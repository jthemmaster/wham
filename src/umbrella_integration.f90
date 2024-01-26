! **********************************************************************
! ** This program does umbrella integration, numerically and ana-     **
! ** lytically (using Gaussian distributions), and WHAM analysis.     **
! **                                                                  **
! ** The methods are described in                                     **
! ** "Bridging the gap between thermodynamic integration and umbrella **
! **  sampling provides a novel analysis method: `Umbrella            **
! **  Integration'", J. Kaestner and W. Thiel, J. Chem. Phys.         **
! **  123, 144104 (2005)                                              **
! ** and                                                              **
! ** "Analysis of the statistical error in umbrella sampling          **
! **  simulations by umbrella integration" Johannes Kästner, Walter   **
! **  Thiel, J. Chem. Phys. 124, 234106 (2006)                        **
! **                                                                  **
! ** The results of the papers above are obtained with this code.     **
! **                                                                  **
! ** This code may be used, distributed, and modified. However, it    **
! ** comes AS IS, without any warranty what so ever. If the methods   **
! ** umbrella integration or WHAM are used in scientific work that    **
! ** results in a publication, it is expected that the corresponding  **
! ** papers are cited.                                                **
! **                                                                  **
! ******************************* Written by Johannes Kaestner 2005 ****
PROGRAM UMBRELLA_INTEGRATION
  IMPLICIT NONE
! **********************************************************************
  INTEGER                    :: NWINDOW,IWINDOW
  INTEGER                    :: NBIN,IBIN
  CHARACTER(256)             :: DIR,LINE
  CHARACTER(256),ALLOCATABLE :: FNAME(:)      ! NWINDOW
  REAL(8)                    :: XMAX,XMIN,TEMPERATURE
  INTEGER       ,ALLOCATABLE :: NDELTA(:,:)   ! NWINDOW,NBIN
  INTEGER       ,ALLOCATABLE :: N(:)          ! NWINDOW
  REAL(8)       ,ALLOCATABLE :: FORCECONST(:) ! NWINDOW
  REAL(8)       ,ALLOCATABLE :: BIAS(:,:)     ! NWINDOW,NBIN
  REAL(8)       ,ALLOCATABLE :: XII(:)        ! NWINDOW
  REAL(8)       ,ALLOCATABLE :: BIN_XI(:)     ! NBIN
  REAL(8)       ,ALLOCATABLE :: WINDOWMEAN(:) ! NWINDOW
  REAL(8)       ,ALLOCATABLE :: WINDOWVAR(:)  ! NWINDOW
  INTEGER                    :: IVAR
  REAL(8)                    :: SVAR,BETA,DER,PI,SVAR2
  INTEGER       ,PARAMETER   :: NVALMAX=100000
  REAL(8)                    :: VALUE(NVALMAX) ! XI OF ONE WIN
  REAL(8)                    :: SLOPEDS,SLOPEMEAN,VARMEAN
  INTEGER                    :: VERBOSE,STARTSKIP,SAMPLESPERWINDOW,ISTART
  INTEGER                    :: IVAL,WINDATA
  CHARACTER(40)              :: EUNIT,EUNITP,str20
  REAL(8)                    :: TOL_WHAM,xi
  LOGICAL(4)                 :: TUI,TWHAM,TDAIDXI
  LOGICAL                    :: TWARNEDL,TWARNEDH
  INTEGER                    :: SEGMENTWIDTH

! GLOBAL ARRAYS USED FOR ERROR BAR ESTIMATION
  REAL(8)       ,ALLOCATABLE :: FE(:)            ! NBIN
  REAL(8)       ,ALLOCATABLE :: DADXI(:)         ! NBIN
  REAL(8)       ,ALLOCATABLE :: VARDADXI(:)      ! NBIN
  REAL(8)       ,ALLOCATABLE :: VARWINDOWMEAN(:) ! NWINDOW
  REAL(8)       ,ALLOCATABLE :: VARWINDOWVAR(:)  ! NWINDOW
  INTEGER       ,PARAMETER   :: MAXEXTR=100
  INTEGER                    :: IEXTR,JEXTR,NEXTR,FEXTR,EXTRBIN(MAXEXTR)
  INTEGER                    :: CURVEXTR(MAXEXTR)! 1 FOR MINIMUM, -1 FOR MAXIMUM
  REAL(8)                    :: EXTRXI(MAXEXTR)
  REAL(8)                    :: COVARIANCE_WIDTH,DELTAA,VARDELTAA
  REAL(8)                    :: CONVINT
  REAL(8)                    :: EXTRRANGE
  INTEGER                    :: KEXTR,BEST
  REAL(8)                    :: BESTFE
  real(8)                    :: kappa,delta
! **********************************************************************
  PI=4.D0*DATAN(1.D0)

! ====================================================================
! == Print header
! ====================================================================
  write(*,'(a)') "************************************************************************"
  write(*,'(a)') "**               Umbrella Integration and WHAM analysis               **"
  write(*,'(a)') "************************************************************************"
! ====================================================================
! == DEFAULTS AND COMMAND LINE ARGUMENTS
! ====================================================================
  TEMPERATURE = 300.0D0
  VERBOSE = 1
  DIR='data'
  XMIN=-2.D0
  XMAX=2.D0
  NBIN=200
  EUNIT="AU"
  startskip=0
  SAMPLESPERWINDOW=-1
  tui=.true.
  twham=.true.
  tdaidxi=.false.
  windata=-1
  segmentwidth=100
  extrrange=0.4D0
  CALL READARGS(VERBOSE,TEMPERATURE,DIR,XMIN,XMAX,NBIN,EUNIT, &
      STARTSKIP,SAMPLESPERWINDOW,tui,twham,windata,tdaidxi, &
      segmentwidth,extrrange)

  call report_r(6,"Temperature",temperature,"K")
  call report_r(6,"Minimum value of the reaction coordinate",xmin,"")
  call report_r(6,"Maximum value of the reaction coordinate",xmax,"")
  call report_i(6,"Number of bins",nbin,"")
  CALL report_string(6,"Input files read from directory",dir)
  call report_i(6,"Number of data lines skipped at start",STARTSKIP,"")
  if(SAMPLESPERWINDOW.gt.0) then
    call report_i(6,"Maximum number of data lines to be used per window",SAMPLESPERWINDOW,"")
  else
    call report_string(6,"Maximum number of data lines to be used per window","all")
  end if
  if(tui) then
    call report_i(6,"Number of data points to be grouped into a segment",segmentwidth,"")
    call report_r(6,"Range for collecting extrema",extrrange,"")
  end if

! determine the energy unit
  call upcase(eunit)
  if(eunit.eq.'AU') then
    eunitp="Hartree"
    BETA=1.D0/(TEMPERATURE*3.166815208D-6)
    tol_wham=1.D-9
  else if(eunit.eq."KCAL") then
    eunitp="kcal/mol"
    BETA=1.D0/(TEMPERATURE*1.982923700D-3)
    tol_wham=1.D-9/627.51D0
  else if(eunit.eq."KJ") then
    eunitp='kJ/mol'
    beta=1.D0/(TEMPERATURE*8.31451D-3)
    tol_wham=1.D-9/2625.5D0
  else
    PRINT*,"ERROR: Unknown energy unit ",trim(eunit)
    call exit(1)
  end if
  CALL report_string(6,"Energy unit in input bias and output",eunitp)
  if(verbose.gt.1) then
    line='1/'//trim(eunitp)
    call report_r(6,"Beta",beta,trim(line))
  end if

! ====================================================================
! == GET INPUT FILE NAMES
! ====================================================================
  nwindow=0
  line="\ls -1 "//trim(dir)//" > .tmp"
  call system(line)
  call system("wc -l .tmp > .tmp2")
  open(unit=10,file=".tmp2")
  READ(10,*) NWINDOW
  CLOSE(10)
  IF(NWINDOW.LT.1) THEN
    PRINT*,"ERROR: NO INPUT FILES FOUND IN DIRECTORY ",TRIM(DIR)
    call system("rm .tmp2 .tmp")
    CALL EXIT(1)
  END IF

  ALLOCATE(FNAME(NWINDOW))
  ALLOCATE(NDELTA(NWINDOW,NBIN))
  ALLOCATE(N(NWINDOW))
  ALLOCATE(FORCECONST(NWINDOW))
  ALLOCATE(XII(NWINDOW))
  ALLOCATE(BIN_XI(NBIN))
  ALLOCATE(BIAS(NWINDOW,NBIN))
  ALLOCATE(WINDOWMEAN(NWINDOW))
  ALLOCATE(WINDOWVAR(NWINDOW))
  ALLOCATE(varWINDOWMEAN(NWINDOW))
  ALLOCATE(varWINDOWVAR(NWINDOW))

  OPEN(UNIT=10,FILE=".tmp")
  DO IWINDOW=1,NWINDOW
    READ(10,*) FNAME(IWINDOW)
  END DO
  CLOSE(10)
  call system("rm .tmp2 .tmp")
  call report_i(6,"Number of windows (input files)",nwindow,"")

! ====================================================================
! == SET MIDDLE OF THE BINS
! ====================================================================
  DO IBIN=1,NBIN
    BIN_XI(IBIN)=XMIN+(REAL(IBIN)-0.5)*(XMAX-XMIN)/DBLE(NBIN)
  END DO

! ====================================================================
! == READ THE DATA AND COLLECT THEM INTO BINS
! ====================================================================
  N=0
  NDELTA=0
  TWARNEDL=.FALSE.
  TWARNEDH=.FALSE.
  DO IWINDOW=1,NWINDOW
    LINE=TRIM(DIR)//'/'//TRIM(FNAME(IWINDOW))
    OPEN(UNIT=10,FILE=TRIM(LINE))
    DO
      READ(10,'(A)',ERR=200,END=200) LINE
      READ(LINE,ERR=201,FMT=*) SVAR,FORCECONST(IWINDOW),XII(IWINDOW)
      ! READ THE DATA INTO ARRAY VALUE
      N(IWINDOW)=N(IWINDOW)+1
      IF(N(IWINDOW).LE.NVALMAX) THEN
        VALUE(N(IWINDOW))=SVAR
      ELSE
        STOP "ERROR: INCREASE NVALMAX"
      END IF
    end DO 
201 print*,"ERROR reading data file ",TRIM(DIR)//'/'//TRIM(FNAME(IWINDOW))
    call exit(1)
200 CONTINUE

    ! HERE, N IS THE NUMBER OF THE LAST ENTRY IN VALUE
    istart=1+startskip
    if(SAMPLESPERWINDOW.gt.0) then
      if(N(iwindow)-startskip.gt.SAMPLESPERWINDOW) istart=N(iwindow)-SAMPLESPERWINDOW+1
    end if
    if(verbose.gt.1) then
      WRITE(*,'("Window "i2,1x,a15," Total values ",i6," Values used from",i6," Number of used values",i6)') &
          IWINDOW,TRIM(FNAME(IWINDOW)),N(IWINDOW),istart,N(IWINDOW)-istart+1
    end if
    
    DO IVAL=ISTART,N(IWINDOW)
      ! BIN THE VALUE
      ! INT ROUNDS TO THE LOWER VALUE      
      SVAR=(VALUE(IVAL)-XMIN)/(XMAX-XMIN)*DBLE(NBIN)
      IF(SVAR.GT.0.D0) THEN
        IVAR=INT(SVAR)+1
        IF(IVAR.LE.NBIN) THEN
          NDELTA(IWINDOW,IVAR)=NDELTA(IWINDOW,IVAR)+1
        ELSE
          IF(.NOT.TWARNEDH.AND.TWHAM) THEN
            PRINT*,"WARNING: VALUES HIGHER THAN RC-RANGE. THIS MAY SPOIL WHAM!"
            TWARNEDH=.TRUE.
          END IF
        ENDIF
      ELSE
        IF(.NOT.TWARNEDL.AND.TWHAM) THEN
          PRINT*,"WARNING: VALUES LOWER THAN RC-RANGE. THIS MAY SPOIL WHAM!"
          TWARNEDL=.TRUE.
        END IF
      END IF
    END DO ! READ
    CLOSE(10)

!   ====================================================================
!   == CALCULATE WINDOW MEAN AND VARIANCE AND THEIR ERROR BARS
!   ====================================================================
    SVAR=DBLE(N(IWINDOW)-ISTART+1)
    WINDOWMEAN(IWINDOW)=SUM(VALUE(ISTART:N(IWINDOW)))/SVAR
    !SIGMA**2 WITH N-1 WEIGHTING
    WINDOWVAR(IWINDOW)=SUM((VALUE(ISTART:N(IWINDOW))-WINDOWMEAN(IWINDOW))**2)&
        /(SVAR-1.D0)
    CALL VAR_COARSE(N(IWINDOW)-ISTART+1,SEGMENTWIDTH,VALUE(ISTART:N(IWINDOW)), &
        VARWINDOWMEAN(IWINDOW),VARWINDOWVAR(IWINDOW))
    ! FROM NOW ON, N IS THE NUMBER OF VALUES USED PER WINDOW
    N(IWINDOW)=N(IWINDOW)-ISTART+1

!   ====================================================================
!   == WRITE INDIVIDUAL HISTOGRAMS AND THEIR DERIVATIVES
!   ====================================================================
    if(verbose.gt.1) then
      call system("mkdir -p histograms")
      line="histograms/"//trim(fname(iwindow))
      OPEN(UNIT=20,FILE=TRIM(LINE))
      WRITE(20,*) "# HISTOGRAM AND ITS DERIVATIVE OF FILE ", &
          TRIM(FNAME(IWINDOW))
      DO IBIN=1,NBIN
        IF(IBIN.LT.NBIN) THEN
          DER=(DBLE(NDELTA(IWINDOW,IBIN+1)-NDELTA(IWINDOW,IBIN)))/&
              (BIN_XI(IBIN+1)-BIN_XI(IBIN))
        ELSE
          DER=0.D0
        END IF
        WRITE(20,*) BIN_XI(IBIN),NDELTA(IWINDOW,IBIN),DER
      END DO
      
      ! ALSO WRITE NORMAL DISTRIBUTION
      WRITE(20,*)
      WRITE(20,*) "# FITTED NORMAL DISTRIBUTION AND ITS DERIVATIVE OF FILE ",&
          TRIM(FNAME(IWINDOW))
      ! PRE-FACTOR OF NORMAL DISTR.
      SVAR=1.D0/DSQRT(WINDOWVAR(IWINDOW)*2.D0*PI)* &
          DBLE(N(IWINDOW))*(BIN_XI(2)-BIN_XI(1))!*DBLE(NBIN)
      DO IBIN=1,NBIN
        ! NORMAL DISTRIBUTION
        SVAR2=SVAR*EXP(-(BIN_XI(IBIN)-WINDOWMEAN(IWINDOW))**2/ &
            (2.D0*WINDOWVAR(IWINDOW)))
        ! DERIVATIVE
        DER=SVAR2*(-(BIN_XI(IBIN)-WINDOWMEAN(IWINDOW))/WINDOWVAR(IWINDOW))
        WRITE(20,*) BIN_XI(IBIN),SVAR2,DER
      END DO
      CLOSE(20)
    end if
    if(verbose.gt.1) &
        write(*,'(10x," Mean ",f15.10," Var ",f15.10)') & 
        windowmean(iwindow),windowvar(iwindow)

  END DO ! NWINDOW

  if(windata.gt.nwindow) windata=-1
  if(windata.gt.0.and.tui) then
    write(*,"('Contributions in UI are written for window ',i2,', ',a)") &
        windata,trim(fname(windata))
    write(str20,*) windata
    str20="window_"//trim(adjustl(str20))
    open(unit=11,file=trim(str20)//"_distributions.xy")
  end if

  SVAR=DBLE(SUM(N))/DBLE(NBIN)
  call report_r(6,"Average number of data points per bin",svar,"")
  
  ! PRINT GLOBAL HISTOGRAM
  if(verbose.gt.1) then
    open(unit=10,file="global_histogram.xy")
    write(10,*) "# Global Histogram, first column represents middle of the bin"
    write(10,*) "# Reaction coord (RCU) frequency (absolute)"  
    DO IBIN=1,NBIN
      xi=BIN_XI(IBIN)
      WRITE(10,*) XI,SUM(NDELTA(:,IBIN))
      if(windata.gt.0.and.tui) then
        write(11,*) XI,NDELTA(windata,IBIN), &
            DBLE(N(WINDATA))*(BIN_XI(2)-bin_xi(1))/DSQRT(WINDOWVAR(WINDATA)*2.D0*PI)* &
            EXP(-(XI-WINDOWMEAN(WINDATA))**2/(2.D0*WINDOWVAR(WINDATA)))
      end if
    END DO
    CLOSE(10)
  end if

  if(windata.gt.0.and.tui) close(11)
  
  ! CALCULATE BIAS ARRAY
  DO IBIN=1,NBIN
    DO IWINDOW=1,NWINDOW
      ! the input force constant corresponds to w = K/2 (x-x0)**2
      ! the wham routine needs k=K/2
      BIAS(IWINDOW,IBIN)=0.5D0*FORCECONST(IWINDOW)* &
          (BIN_XI(IBIN)-XII(IWINDOW))**2
    END DO
  END DO

  if(twham) then
    CALL WHAM(NBIN,NWINDOW,NDELTA,N,BIN_XI,BETA,BIAS,verbose, &
        tol_wham,eunitp)
  end if

  if(verbose.gt.2) then
    ! ====================================================================
    ! == Calculate the slope of DA/DXI and compare it to the slope
    ! == from averaging the neighbor mean values
    ! ====================================================================
    print*,"Negative diff means, variance should be smaller"
    print*,"  bin,      slope(UI) slope(FS)     diff     var     var(FS)"
    DO IWINDOW=2,NWINDOW-1
      SLOPEDS=1.D0/BETA/WINDOWVAR(IWINDOW)-FORCECONST(IWINDOW)
      SVAR=-FORCECONST(IWINDOW+1)*(WINDOWMEAN(IWINDOW+1)-XII(IWINDOW+1))
      SVAR=SVAR+FORCECONST(IWINDOW-1)*(WINDOWMEAN(IWINDOW-1)-XII(IWINDOW-1))
      SLOPEMEAN=SVAR/(WINDOWMEAN(IWINDOW+1)-WINDOWMEAN(IWINDOW-1))
      VARMEAN=1.D0/(BETA*(SLOPEMEAN+FORCECONST(IWINDOW)))
      WRITE(*,'(I2,2X,A7,5F10.5)') IWINDOW,TRIM(FNAME(IWINDOW)), &
          SLOPEDS,SLOPEMEAN,SLOPEDS-SLOPEMEAN,WINDOWVAR(IWINDOW),VARMEAN
    END DO
  end if

  if(tui) then
    allocate(fe(nbin))
    allocate(dadxi(nbin))
    allocate(vardadxi(nbin))
    CALL UMBRELLA_INTEGRATION_ANALYTIC(NBIN,NWINDOW,NDELTA,N,BIN_XI, &
        BETA,FORCECONST,XII,WINDOWMEAN,WINDOWVAR,varWINDOWMEAN,varWINDOWVAR, &
        verbose,eunitp,windata,fe,dadxi,vardadxi)
    ! ==================================================================
    ! == Recommendations for parameter choice
    ! ==================================================================
    write(*,'(a)') "Recommendations for the choice of future sampling parameters:"
    kappa=0.D0
    DELTA=BIN_XI(2)-BIN_XI(1)
    DO IBIN=9,NBIN-7,2
!      if(dble(abs(nbin/2-ibin))/dble(nbin).gt.0.4D0) cycle
      ! second derivative from nearest neighbours
      svar=(fe(ibin-2)-2.D0*fe(ibin)+fe(ibin+2))/(2.D0*delta)**2
      ! from further away
      svar2=(fe(ibin-6)-2.D0*fe(ibin)+fe(ibin+6))/(6.D0*delta)**2
      ! check similarity of these and dischard too noisy values
      if(abs(svar2/svar).gt.1.5D0.or.abs(svar/svar2).GT.1.5D0) cycle
      ! average 3 different estimates of the second derivative
      svar=svar+svar2+(fe(ibin-4)-2.D0*fe(ibin)+fe(ibin+4))/(4.D0*delta)**2
      svar=svar/3.D0

      if(svar.lt.kappa) then
        kappa=svar
      end if
    end DO
    kappa=-kappa
    str20=trim(eunitp)//" * RCU**-2"
    CALL report_r(6,"Kappa",kappa,str20)
    write(*,'("Rough guess: force constant should be 3-10 times kappa, thus",f10.5," to ",f10.5)') &
        3.D0*kappa,10.D0*kappa
    CALL report_r(6,"Force constant is",FORCECONST(1),str20)
    CALL report_r(6,"Average window spacing should be smaller than", &
        3.D0/sqrt(beta*FORCECONST(1)),"RCU")
    call report_r(6,"Average window spacing is",(xii(nwindow)-xii(1))/dble(nwindow-1),"RCU")
    call report_r(6,"Minimum number of equally-spaced windows",&
        (xii(nwindow)-xii(1))*sqrt(beta*FORCECONST(1))/3.D0+1.D0," (round!)")
    write(*,*)
  end if

  ! ====================================================================
  ! == ERROR BAR ANALYSIS
  ! ====================================================================
  IF(TUI) THEN

    ! PLOT ERRORS IN THE DERIVATIVE
    OPEN(UNIT=10,FILE='vardadxi.xy')
    DO IBIN=1,NBIN
      WRITE(10,*) BIN_XI(IBIN), VARDADXI(IBIN)
    END DO
    CLOSE(10)

    ! FIND EXTREMA FOR THIS SIMULATION
    IEXTR=0
    DO IBIN=1,NBIN-1
      IF(DADXI(IBIN)*DADXI(IBIN+1).LT.0.D0) THEN
        ! WE HAVE AN EXTREMUM
        IEXTR=IEXTR+1
        IF(IEXTR.GT.MAXEXTR) STOP "INCREASE MAXEXTR"

        EXTRBIN(IEXTR)=IBIN
        EXTRXI(IEXTR)=BIN_XI(IBIN)-DADXI(IBIN)*(BIN_XI(IBIN+1)-BIN_XI(IBIN))/ &
            (DADXI(IBIN+1)-DADXI(IBIN))
        IF(DADXI(IBIN).GT.0.D0) THEN
          curvextr(iextr)=-1
          if(verbose.gt.1) write(*,"('Maximum',i3,' bin',i5,' RC',f15.10)") &
              iextr,extrbin(iextr),extrxi(iextr)
        else
          curvextr(iextr)=1
          if(verbose.gt.1) write(*,"('Minimum',i3,' bin',i5,' RC',f15.10)") &
              iextr,extrbin(iextr),extrxi(iextr)
        END IF
      end IF
    end DO
    nextr=iextr

    ! CLUSTER EXTREMA AND FIND SIGNIFICANT ONES
    IEXTR=1
    fextr=0
    DO
      do jextr=iextr+1,nextr+1
        if( (jextr.gt.nextr).or. &
            ((extrxi(jextr)-extrxi(iextr)).gt.extrrange)) then
          fextr=fextr+1
          if(verbose.gt.1) write(*,'("Extrema",i3," -",i3," are collected to",i3)') &
              iextr,jextr-1,fextr
          ! find best extremum 
          ivar=sum(curvextr(iextr:jextr-1))
          if(ivar.eq.0) then
            print*,"ERROR: Try a different value of -r"
            call exit(1)
          end if
          bestfe=fe(extrbin(iextr))
          best=iextr
          do kextr=iextr+1,jextr-1
            if(fe(extrbin(kextr))*dble(ivar).lt.bestfe*dble(ivar)) then
              bestfe=fe(extrbin(kextr))
              best=kextr
            end if
          end do
          EXTRBIN(fextr)=extrbin(best)
          EXTRXI(FEXTR)=EXTRXI(best)

          iextr=jextr
          exit
        end if
      end do
      if(iextr.gt.nextr) exit
    end DO
    nextr=fextr
    call report_i(6,"Number of extrema",nextr,"")

    do iextr=1,nextr
      IF(curvextr(iextr).eq.1) then
        line="Minimum"
      else
        line="Maximum"
      end IF
      WRITE(*,'(A,1X,I3," RC=",F10.5)') TRIM(LINE),IEXTR,EXTRXI(IEXTR)
    END DO

    COVARIANCE_WIDTH=SUM(SQRT(WINDOWVAR))/DBLE(NWINDOW)
    DO IEXTR=1,NEXTR
      DO JEXTR=IEXTR+1,NEXTR

        CALL INTEGRATE_SIMPSON(EXTRXI(IEXTR),EXTRXI(JEXTR),NBIN,BIN_XI, &
            DADXI,DELTAA)

        if((EXTRXI(JEXTR)-EXTRXI(IEXTR)).gt.COVARIANCE_WIDTH) then
          ! ESTIMATE THE ERROR BAR CONSIDERING CORRELATION
          VARDELTAA=SUM(VARDADXI(EXTRBIN(IEXTR):EXTRBIN(JEXTR))) / &
              DBLE(EXTRBIN(JEXTR)-EXTRBIN(IEXTR)+1)
          VARDELTAA=VARDELTAA * ( (EXTRXI(JEXTR)-EXTRXI(IEXTR)) * &
              COVARIANCE_WIDTH*SQRT(2.D0*PI) - &
              2.D0*COVARIANCE_WIDTH**2 )
          CONVINT=1.96D0*SQRT(VARDELTAA)
        
          ! PRINT RESULT
          write(*,'("Energy difference extrema ",i3," -",i3," = ",f15.10," +/- ",f15.10,1x,a)') &
              JEXTR,IEXTR,DELTAA,CONVINT,TRIM(EUNITP)
        else
          write(*,'("Energy difference extrema ",i3," -",i3," = ",f15.10,1x,a)') &
              JEXTR,IEXTR,DELTAA,TRIM(EUNITP)
        end if

      END DO
    END DO
  END IF

!!$  CALL FORCE_SAMPLING(NBIN,NWINDOW,NDELTA,N,BIN_XI,BETA,FORCECONST,&
!!$      XII,WINDOWMEAN,WINDOWVAR,verbose)
!!$

  if(tdaidxi) then
    CALL PLOT_DAIDXI(NBIN,NWINDOW,BIN_XI,BETA,FORCECONST,&
        XII,WINDOWMEAN,WINDOWVAR,eunitp)
  end if

!!$
!!$  CALL UMBRELLA_INTEGRATION_NUMERIC(NBIN,NWINDOW,NDELTA,N,BIN_XI, &
!!$      BETA,FORCECONST,XII,WINDOWMEAN,WINDOWVAR,verbose)
!!$
! ====================================================================
! == TRY WHAM WITH SMOOTHENED DISTRIBUTIONS
! ====================================================================

!!$  ! K=2K PROBLEM!
!!$  FORCECONST=FORCECONST/2.D0
!!$  ALLOCATE(NDELTA_r(NWINDOW,NBIN))
!!$
!!$  DO IWINDOW=1,NWINDOW
!!$    ! PRE-FACTOR OF NORMAL DISTR.
!!$    SVAR=1.D0/DSQRT(WINDOWVAR(IWINDOW)*2.D0*PI)* &
!!$        DBLE(N(IWINDOW))*(BIN_XI(2)-BIN_XI(1))
!!$    DO IBIN=1,NBIN
!!$      SVAR2=SVAR*EXP(-(BIN_XI(IBIN)-WINDOWMEAN(IWINDOW))**2/ &
!!$          (2.D0*WINDOWVAR(IWINDOW)))
!!$      NDELTA_r(IWINDOW,IBIN)=SVAR2*dble(N(IWINDOW))
!!$    END DO
!!$  END DO
!!$  CALL WHAM_real(NBIN,NWINDOW,NDELTA_r,N,BIN_XI,BETA,BIAS,verbose)

END PROGRAM UMBRELLA_INTEGRATION
!
! ......................................................................
SUBROUTINE UMBRELLA_INTEGRATION_ANALYTIC(NBIN,NWINDOW,NDELTA,N,BIN_XI, &
    BETA,FORCECONST,XII,WINDOWMEAN,WINDOWVAR,VARWINDOWMEAN,VARWINDOWVAR, &
    VERBOSE,EUNITP,WINDATA,FE,DADXI,VARDADXI)
! **********************************************************************
! ** This routine does umbrella integration with both the derivative  **
! ** and the weights calculated from the normal distribution. The     **
! ** function A(xi) is then obtained by integration with Simpson's    **
! ** rule.                                                            **
! ** FE contains the free energy. Only 3,nbin,2 values are integrated **
! ** the others are interpolated. Only 3,nbin-1 are set.              **
! **********************************************************************
  IMPLICIT NONE
  INTEGER      ,INTENT(IN) :: NBIN
  INTEGER      ,INTENT(IN) :: NWINDOW
  INTEGER      ,INTENT(IN) :: NDELTA(NWINDOW,NBIN)
  INTEGER      ,INTENT(IN) :: N(NWINDOW)
  REAL(8)      ,INTENT(IN) :: BIN_XI(NBIN)
  REAL(8)      ,INTENT(IN) :: BETA  
  REAL(8)      ,INTENT(IN) :: FORCECONST(NWINDOW)
  REAL(8)      ,INTENT(IN) :: XII(NWINDOW)
  REAL(8)      ,INTENT(IN) :: WINDOWMEAN(NWINDOW)
  REAL(8)      ,INTENT(IN) :: WINDOWVAR(NWINDOW)
  REAL(8)      ,INTENT(IN) :: VARWINDOWMEAN(NWINDOW)
  REAL(8)      ,INTENT(IN) :: VARWINDOWVAR(NWINDOW)
  INTEGER      ,INTENT(IN) :: VERBOSE
  CHARACTER(*) ,INTENT(IN) :: EUNITP
  INTEGER      ,INTENT(IN) :: WINDATA
  REAL(8)      ,INTENT(OUT):: FE(NBIN)
  REAL(8)      ,INTENT(OUT):: DADXI(NBIN)
  REAL(8)      ,INTENT(OUT):: VARDADXI(NBIN)
  INTEGER                  :: IBIN,IWINDOW
  REAL(8)                  :: SVAR
  REAL(8)                  :: XI,PI,daidxi
  REAL(8)                  :: DELTA,Z
  REAL(8)                  :: WEIGHT,TOTWEIGHT
  REAL(8)                  :: LIN,QUAD,RES
  REAL(8)                  :: VARDAUIDXI
  CHARACTER(20)            :: STR20
! **********************************************************************
  PI=4.D0*DATAN(1.D0)

  if(windata.gt.0) then
    write(str20,*) windata
    str20="window_"//trim(adjustl(str20))
    open(unit=11,file=trim(str20)//"_weight.xy")
    write(11,*) "# Reaction coordinate (RCU), weight to global average"
    open(unit=12,file=trim(str20)//"_contributions.xy")
    write(12,*) "# Contributions. The columns have the following meaning:"
    write(12,*) "# Reaction coordinate (RCU)"
    write(12,*) "# Unbiased A(xi) (",trim(eunitp),")"
    write(12,*) "# Linear contribution to A(xi) (",trim(eunitp),")"
    write(12,*) "# Quadratic contribution to A(xi) (",trim(eunitp),")"
    write(12,*) "# Residuum x 10 (",trim(eunitp),")"
  end if
  
! ====================================================================
! == CALCULATE DA/DXI
! ====================================================================
  DADXI=0.D0
  vardadxi=0.D0
  DO IBIN=1,NBIN
    TOTWEIGHT=0.D0
    DO IWINDOW=1,NWINDOW
        
        XI=BIN_XI(IBIN)
        DAIDXI=1.D0/BETA*(XI-WINDOWMEAN(IWINDOW))/WINDOWVAR(IWINDOW)- &
            FORCECONST(IWINDOW)*(XI-XII(IWINDOW))

        ! VARIANCE
        VARDAUIDXI=(VARWINDOWMEAN(IWINDOW) + &
            ( (XI-WINDOWMEAN(IWINDOW)) / WINDOWVAR(IWINDOW) ) **2 * &
            VARWINDOWVAR(IWINDOW) ) / ( BETA*WINDOWVAR(IWINDOW) )**2

        ! WEIGHT ACCORDING TO N(IBIN)*Z(XI)
        Z=1.D0/DSQRT(WINDOWVAR(IWINDOW)*2.D0*PI)* &
            EXP(-(XI-WINDOWMEAN(IWINDOW))**2/(2.D0*WINDOWVAR(IWINDOW)))
        WEIGHT=DBLE(N(IWINDOW))*Z 
        TOTWEIGHT=TOTWEIGHT+WEIGHT

        ! ADD WEIGHTED CONTRIBUTION
        DADXI(IBIN)=DADXI(IBIN)+daidxi*WEIGHT
        vardadxi(ibin)=vardadxi(ibin)+vardauidxi * weight**2
    END DO
! ====================================================================
! == PLOT DAI/DXI AND ITS CONTRIBUTIONS FOR ONE PARTICULAR WINDOW
! ====================================================================
    IF(TOTWEIGHT.GT.10.D0) THEN
      DADXI(IBIN)=DADXI(IBIN)/TOTWEIGHT
      vardadxi(ibin)=vardadxi(ibin) / TOTWEIGHT**2

      IF(windata.gt.0) then
        ! WRITE CONTRIBUTIONS FOR ONE PARTICULAR WINDOW
        IWINDOW=WINDATA
        Z=1.D0/DSQRT(WINDOWVAR(IWINDOW)*2.D0*PI)* &
            EXP(-(XI-WINDOWMEAN(IWINDOW))**2/(2.D0*WINDOWVAR(IWINDOW)))
        WRITE(11,*) XI,DBLE(N(IWINDOW))*Z/TOTWEIGHT
        IF(NDELTA(IWINDOW,IBIN).GT.0) THEN
          ! center unbiased data
          svar=DBLE(N(IWINDOW))/dble(maxval(NDELTA(IWINDOW,:)))
          ! UNBIASED RAW DATA
          SVAR=-LOG(svar*DBLE(NDELTA(IWINDOW,IBIN))/DBLE(N(IWINDOW)))/beta
          SVAR=SVAR-0.5D0*FORCECONST(IWINDOW)*(BIN_XI(IBIN)- &
              XII(IWINDOW))**2
          
          LIN=FORCECONST(IWINDOW) * ( XII(IWINDOW) - WINDOWMEAN(IWINDOW) )
          QUAD=0.5D0*(1.D0/BETA/WINDOWVAR(IWINDOW)-FORCECONST(IWINDOW))
          
          LIN = ( XI - WINDOWMEAN(IWINDOW) ) * LIN 
          QUAD = QUAD * ( XI - WINDOWMEAN(IWINDOW) ) **2 
          RES= ( SVAR - LIN - QUAD ) * 10.D0
          
          WRITE(12,'(6F15.7)') XI,SVAR,LIN,QUAD,RES !,EXACTRES
        END IF
      END IF

    ELSE
      DADXI(IBIN)=0.D0 ! HERE, ONE COULD USE AN INTERPOLATION!
      vardadxi(ibin)=0.D0
    END IF
  END DO
  if(windata.gt.0) THEN
    CLOSE(11)
    CLOSE(12)
  END if

  if(verbose.gt.1) then
    ! PLOT DADXI
    open(unit=10,file="dadxi_norm.xy")
    write(10,*) "# Mean force, dA/dxi in Umbrella Integration"
    write(10,*) "# Reaction coordinate (RCU), dA/dxi (",trim(eunitp),"/RCU)"
    DO IBIN=1,NBIN-1
      WRITE(10,*) BIN_XI(IBIN),DADXI(IBIN)
    END DO
    CLOSE(10)
  end if

! ====================================================================
! == INTEGRATE
! ====================================================================
  DELTA=BIN_XI(2)-BIN_XI(1)
  fe=0.d0
  DO IBIN=3,NBIN,2
    fe(ibin)=fe(ibin-2)+DELTA/3.D0*(DADXI(IBIN-2)+DADXI(IBIN-1)*4.D0+DADXI(IBIN))
  END DO

  ! INTERPOLATE THE INTERMEDIATE VALUES OF FE
  DO IBIN=2,NBIN-1,2
    FE(IBIN)=0.5D0*(FE(IBIN-1)+FE(IBIN+1))
  END DO

  open(unit=10,file="fe_ui.xy")
  write(10,*) "# Umbrella integration analysis"
  write(10,*) "# Reaction coordinate (RCU), free energy (",trim(eunitp),")"
  svar=minval(fe(3:nbin-1))
  fe(3:nbin-1)=fe(3:nbin-1)-svar
  DO IBIN=3,NBIN-1
    write(10,*) BIN_XI(IBIN),fe(ibin)
  end DO
  CLOSE(10)
  call report_string(6,"Results of UI (A(RC)) are written to file","fe_ui.xy")

END SUBROUTINE UMBRELLA_INTEGRATION_ANALYTIC
!
!.......................................................................
SUBROUTINE PLOT_DAIDXI(NBIN,NWINDOW,BIN_XI,BETA,&
    FORCECONST,XII,WINDOWMEAN,WINDOWVAR,eunitp)
! **********************************************************************
! ** ONLY A PLOTTING TOOL                                             **
! **********************************************************************
  IMPLICIT NONE
  INTEGER      ,INTENT(IN) :: NBIN
  INTEGER      ,INTENT(IN) :: NWINDOW
  REAL(8)      ,INTENT(IN) :: BIN_XI(NBIN)
  REAL(8)      ,INTENT(IN) :: BETA  
  REAL(8)      ,INTENT(IN) :: FORCECONST(NWINDOW)
  REAL(8)      ,INTENT(IN) :: XII(NWINDOW)
  REAL(8)      ,INTENT(IN) :: WINDOWMEAN(NWINDOW)
  REAL(8)      ,INTENT(IN) :: WINDOWVAR(NWINDOW)
  character(*) ,intent(in) :: eunitp
  INTEGER                  :: IBIN,IWINDOW
  REAL(8)                  :: SVAR
  REAL(8)                  :: XI,PI
  REAL(8)                  :: Z
! **********************************************************************
  PI=4.D0*DATAN(1.D0)
  
! ====================================================================
! == CALCULATE DA/DXI
! ====================================================================
  open(unit=10,file="daidxi.xy")
  write(10,*) "# Mean force of each window, dA_i/dxi in Umbrella Integration"
  write(10,*) "# Reaction coordinate (RCU), dA_i/dxi (",trim(eunitp),"/RCU)"
  DO IWINDOW=1,NWINDOW
    DO IBIN=1,NBIN
!      IF(NDELTA(IWINDOW,IBIN).GT.0) THEN
        
        XI=BIN_XI(IBIN)
        SVAR=1.D0/BETA*(XI-WINDOWMEAN(IWINDOW))/WINDOWVAR(IWINDOW)- &
            FORCECONST(IWINDOW)*(XI-XII(IWINDOW))

        ! WEIGHT ACCORDING TO N(ibin)*Z(xi)
        Z=1.D0/DSQRT(WINDOWVAR(IWINDOW)*2.D0*PI)* &
            EXP(-(XI-WINDOWMEAN(IWINDOW))**2/(2.D0*WINDOWVAR(IWINDOW)))

        if(z.gt.0.1D0) write(10,*) xi,svar
    END DO
    write(10,*)
  END DO
  close(10)
  
  open(unit=10,file="daidxi_ximax.xy")
  write(10,*) "# Mean force at each window mean, dA_i/dxi|xi-bar in Umbrella Integration"
  write(10,*) "# Reaction coordinate (RCU), dA_i/dxi (",trim(eunitp),"/RCU)"
  DO IWINDOW=1,NWINDOW
    write(10,*) WINDOWMEAN(IWINDOW),-FORCECONST(IWINDOW)* &
        (WINDOWMEAN(IWINDOW)-XII(IWINDOW))
  end DO
  close(10)
END SUBROUTINE PLOT_DAIDXI
!
!.......................................................................
SUBROUTINE WHAM(NBIN,NWINDOW,NDELTA,N,BIN_XI,BETA,BIAS,verbose, &
    tol_wham,eunitp)
! **********************************************************************
! ** FOLLOWING ROU95                                                  **
! **********************************************************************
  IMPLICIT NONE
  INTEGER      ,INTENT(IN) :: NBIN
  INTEGER      ,INTENT(IN) :: NWINDOW
  INTEGER      ,INTENT(IN) :: NDELTA(NWINDOW,NBIN)
  INTEGER      ,INTENT(IN) :: N(NWINDOW)
  REAL(8)      ,INTENT(IN) :: BIN_XI(NBIN)
  REAL(8)      ,INTENT(IN) :: BETA  
  REAL(8)      ,INTENT(IN) :: BIAS(NWINDOW,NBIN)
  INTEGER      ,INTENT(IN) :: verbose
  REAL(8)      ,INTENT(IN) :: tol_wham
  character(*) ,intent(in) :: eunitp
!  REAL(8)                  :: TOL=1.D-9
  INTEGER                  :: ITER,NITER=100000,NOM,IBIN,IWINDOW
  LOGICAL                  :: CONV
  REAL(8)                  :: FI(NWINDOW),ZU(NBIN)
  REAL(8)                  :: FIOLD(NWINDOW)
  REAL(8)                  :: SVAR,DENOM,SVAR2
  real(8)                  :: a
! **********************************************************************
  FI=0.D0
  FIOLD=1.D0
  CONV=.FALSE.
  if(verbose.le.1) print*,"WHAM loop running ... "
  DO ITER=1,NITER
    ! EQ 8
    DO IBIN=1,NBIN
      DENOM=0.D0
      DO IWINDOW=1,NWINDOW
        SVAR=EXP(-BETA*(BIAS(IWINDOW,IBIN)-FI(IWINDOW)))
        DENOM=DENOM+SVAR*DBLE(N(IWINDOW))
      END DO
      NOM=0
      DO IWINDOW=1,NWINDOW
        NOM=NOM+NDELTA(IWINDOW,IBIN)
      END DO
      ZU(IBIN)=DBLE(NOM)/DENOM
    END DO
    
    ! EQ 9
    DO IWINDOW=1,NWINDOW
      SVAR=0.D0
      DO IBIN=1,NBIN
        SVAR2=EXP(-BETA*BIAS(IWINDOW,IBIN))
        SVAR=SVAR+SVAR2*ZU(IBIN)
      END DO
      FI(IWINDOW)=-LOG(SVAR)/BETA
    END DO
    
    ! CENTER THE FI AROUND 0
    FI=FI-SUM(FI)/DBLE(NWINDOW)
    
    ! CHECK CONVERGENCE
    SVAR=MAXVAL(ABS(FI-FIOLD))
    CONV=(SVAR.LT.TOL_wham)
    ! check for out-of-bounds
    if(abs(svar).gt.huge(svar)) then
      print*,"ERROR in WHAM loop."
      print*,"Possible reasons: wrong bias, wrong unit of the force constant"
      return
    end if

    ! REPORT
    IF(MOD(ITER,100).EQ.0.and.verbose.gt.1) THEN
      PRINT*,"WHAM ITERATION",ITER,"DIFF",SVAR
    END IF

    IF(CONV) EXIT

    FIOLD=FI
  END DO
  if(verbose.le.1) print*,'done'
  IF(.NOT.CONV) THEN
    WRITE(0,*) "ERROR: WHAM LOOP NOT CONVERGED!"
  ELSE
    CALL REPORT_I(6,"WHAM loop converged after",ITER,"iterations")
  END IF

  ! WRITE THE RESULT TO FILE
  OPEN(UNIT=10,FILE="fe_wham.xy")
  write(10,*) "# WHAM analysis"
  write(10,*) "# Reaction coordinate (RCU), free energy (",trim(eunitp),")"
  !OFFSET
  ! DETERMINE MAX AND MIN
  SVAR=-LOG(MAXVAL(ZU))/BETA
  DO IBIN=1,NBIN
    IF(ZU(IBIN).GT.0.D-30) then
      a=-LOG(ZU(IBIN))/BETA-SVAR
!      WRITE(10,*) BIN_XI(IBIN),a

    end IF
  END DO

  !write
  DO IBIN=1,NBIN
    IF(ZU(IBIN).GT.0.D-30) then
      a=-LOG(ZU(IBIN))/BETA-SVAR
      WRITE(10,*) BIN_XI(IBIN),a
    end IF
  END DO

  CLOSE(10)

  call report_string(6,"Results of WHAM (A(RC)) are written to file","fe_wham.xy")

END SUBROUTINE WHAM
!
! ......................................................................
SUBROUTINE INTEGRATE_SIMPSON(XSTART,XEND,NBIN,X,DERIVATIVE,INTEGRAL)
! **********************************************************************
! ** INTEGRATE AN ARRAY USING SIMPSON'S RULE                          **
! ** GENERGAL PURPOSE ROUTINE                                         **
! **********************************************************************
  IMPLICIT NONE
  REAL(8)      ,INTENT(IN)  :: XSTART
  REAL(8)      ,INTENT(IN)  :: XEND
  INTEGER      ,INTENT(IN)  :: NBIN
  REAL(8)      ,INTENT(IN)  :: X(NBIN)
  REAL(8)      ,INTENT(IN)  :: DERIVATIVE(NBIN)
  REAL(8)      ,INTENT(OUT) :: INTEGRAL
  INTEGER                   :: ISTART,IEND,SSTART,SEND,IBIN
  REAL(8)                   :: DELTA,ADD,SVAR,DSTART,DEND
  REAL(8)                   :: SIMPSON_FACTOR(NBIN) ! 1,4,2,4,2,...,4,1
! **********************************************************************

  !FIND THE POSITIONS OF START AND END OF THE INTEGRAL
  ISTART=-1
  IEND=-1
  DELTA=(X(2)-X(1))
  IF(DELTA.LE.0.D0) STOP "ERROR: X-VALUES NOT MONOTONIC INCREASING!"
  DO IBIN=1,NBIN-1
    IF(ABS(X(IBIN+1)-X(IBIN)-DELTA).GT.1.D-10) &
        STOP "ERROR: NON-EQUIDISTANT X-VALUES"
    IF(XSTART.GE.X(IBIN).AND.XSTART.LT.X(IBIN+1)) THEN
      ISTART=IBIN
    END IF
    IF(XEND.GE.X(IBIN).AND.XEND.LT.X(IBIN+1)) THEN
      IEND=IBIN
    END IF
  END DO
  IF(ISTART.LT.0) STOP "ERROR: START VALUE OUT OF RANGE"
  IF(IEND.LT.0) STOP "ERROR: END VALUE OUT OF RANGE"
!  PRINT*,"ISTART,IEND",ISTART,IEND
  

  IF(MOD(IEND-ISTART,2).EQ.1) THEN
    ! NO CORRCTION NECESSARY
    SSTART=ISTART+1
    SEND=IEND
    ADD=0.D0
  ELSE
    ! LEAVE OUT THE LAST
    SSTART=ISTART+1
    SEND=IEND-1
    ! CORRECT THE LAST BIN (TRAPEZOIDAL RULE)
    ADD=0.5D0*(DERIVATIVE(IEND-1)+DERIVATIVE(IEND))*DELTA
!    PRINT*,"CORRECTION FOR LAST BIN"
  END IF
  IF(SEND.LT.SSTART) STOP "ERROR: RANGE TOO NARROW"
!  PRINT*,SSTART,SEND,SEND-SSTART, "LAST VAL SHOULD BE EVEN"

  ! SIMPSON PART
  INTEGRAL=0.D0

  SIMPSON_FACTOR=0.D0
  DO IBIN=SSTART,SEND-1,2
    SIMPSON_FACTOR(IBIN)=2.D0
    SIMPSON_FACTOR(IBIN+1)=4.D0
  END DO
  SIMPSON_FACTOR(SSTART)=1.D0
  SIMPSON_FACTOR(SEND)=1.D0

  DO IBIN=SSTART,SEND
    INTEGRAL=INTEGRAL+SIMPSON_FACTOR(IBIN)*DERIVATIVE(IBIN)
  END DO
  INTEGRAL=INTEGRAL*DELTA/3.D0
  ! END OF SIMPSON PART

  INTEGRAL=INTEGRAL+ADD

  ! ADD CORRECTION FOR THE ENDS (TRAPEZOIDAL RULE):
  ! DERIVATIVE AT XSTART
  SVAR=(X(ISTART+1)-XSTART)/DELTA ! FRACTION OF THE 2ND VALUE
!  PRINT*,"SVAR",SVAR
  DSTART=SVAR*DERIVATIVE(ISTART)+(1.D0-SVAR)*DERIVATIVE(ISTART+1)
  INTEGRAL=INTEGRAL+0.5D0*(DSTART+DERIVATIVE(ISTART+1))*DELTA*SVAR

  SVAR=(XEND-X(IEND))/DELTA ! FRACTION OF THE 2ND VALUE
!  PRINT*,"SVAR",SVAR
  DEND=(1.D0-SVAR)*DERIVATIVE(IEND)+(SVAR)*DERIVATIVE(IEND+1)
  INTEGRAL=INTEGRAL+0.5D0*(DEND+DERIVATIVE(IEND))*DELTA*SVAR

END SUBROUTINE INTEGRATE_SIMPSON
!
! ......................................................................
SUBROUTINE VAR_COARSE(COUNT,SEGMENTWIDTH,XIARR,VARMEAN,VARVAR)
! **********************************************************************
! ** Read a trajectory of values and calculate the variance of their  **
! ** mean and the variance of the variance using coarse-graining.     **
! **********************************************************************
  IMPLICIT NONE
  INTEGER      ,INTENT(IN)  :: COUNT
  INTEGER      ,INTENT(IN)  :: SEGMENTWIDTH
  REAL(8)      ,INTENT(IN)  :: XIARR(COUNT)
  REAL(8)      ,INTENT(OUT) :: VARMEAN,VARVAR
  REAL(8)                   :: MEAN
  REAL(8)                   :: VARSEGMENTS
  INTEGER                   :: SEGMENTSTART,SEGMENTEND,NSEGMENT,ISEGMENT
  REAL(8)   ,ALLOCATABLE    :: SEGMENTMEAN(:),SEGMENTVAR(:) ! NSEGMENT
! **********************************************************************
  MEAN=SUM(XIARR(1:COUNT))/DBLE(COUNT)
  NSEGMENT=COUNT/SEGMENTWIDTH
  ALLOCATE(SEGMENTMEAN(NSEGMENT))
  ALLOCATE(SEGMENTVAR(NSEGMENT))
  IF(NSEGMENT*SEGMENTWIDTH.NE.COUNT) THEN
    PRINT*,"WARNING, VALUES REMAIN IN SEGMENTING"
  END IF
  SEGMENTMEAN=0.D0
  SEGMENTVAR=0.D0
  DO ISEGMENT=1,NSEGMENT
    SEGMENTSTART=(ISEGMENT-1)*SEGMENTWIDTH+1 ! 1-100, 101-200, ...
    SEGMENTEND=ISEGMENT*SEGMENTWIDTH
    SEGMENTMEAN(ISEGMENT)=SUM(XIARR(SEGMENTSTART:SEGMENTEND))/ &
        DBLE(SEGMENTWIDTH)
    SEGMENTVAR(ISEGMENT)=SUM((XIARR(SEGMENTSTART:SEGMENTEND)- &
        SEGMENTMEAN(ISEGMENT))**2) / DBLE(SEGMENTWIDTH-1)
  END DO
  VARMEAN= SUM((SEGMENTMEAN(1:NSEGMENT)-MEAN)**2) / & 
      DBLE(NSEGMENT-1) / DBLE(NSEGMENT)
  VARSEGMENTS=SUM(SEGMENTVAR(1:NSEGMENT))/DBLE(NSEGMENT)
  VARVAR=SUM((SEGMENTVAR(1:NSEGMENT)-VARSEGMENTS)**2)/ & 
      DBLE(NSEGMENT-1)/DBLE(NSEGMENT)
  DEALLOCATE(SEGMENTMEAN)
  DEALLOCATE(SEGMENTVAR)
END SUBROUTINE VAR_COARSE

! ......................................................................
SUBROUTINE UPCASE(STRING)
  IMPLICIT NONE
  CHARACTER(*),INTENT(INOUT)  :: STRING
  CHARACTER(1)                :: CHA
  INTEGER                     :: I
  DO I=1,LEN(STRING)
     CHA=STRING(I:I)
     IF(ICHAR(CHA).GE.97.AND.ICHAR(CHA).LE.122) THEN
        STRING(I:I)=ACHAR(ICHAR(CHA)-32)
     END IF
  END DO
END SUBROUTINE UPCASE

!==============================================================================
! Report routines 
!==============================================================================
! ......................................................................
SUBROUTINE REPORT_I(NFIL,NAME,VALUE,UNIT)
  INTEGER     ,INTENT(IN) :: NFIL
  CHARACTER(*),INTENT(IN) :: NAME
  INTEGER     ,INTENT(IN) :: VALUE
  CHARACTER(*),INTENT(IN) :: UNIT
  WRITE(NFIL,FMT='(55("."),": ",T1,A,T58,I15," ",A)')NAME,VALUE,UNIT
end SUBROUTINE REPORT_I

! ......................................................................
SUBROUTINE REPORT_R(NFIL,NAME,VALUE,UNIT)
  INTEGER     ,INTENT(IN) :: NFIL
  CHARACTER(*),INTENT(IN) :: NAME
  REAL(8)     ,INTENT(IN) :: VALUE
  CHARACTER(*),INTENT(IN) :: UNIT
  IF(DABS(VALUE).LT.1.D+4.AND.DABS(VALUE).GT.1.D-3) THEN
     WRITE(NFIL,FMT='(55("."),": ",T1,A,T58,F15.10," ",A)')NAME,VALUE,UNIT
  ELSE
     WRITE(NFIL,FMT='(55("."),": ",T1,A,T58,ES15.9," ",A)')NAME,VALUE,UNIT
  END IF
END SUBROUTINE REPORT_R

! ......................................................................
SUBROUTINE REPORT_string(NFIL,NAME,str)
  INTEGER     ,INTENT(IN) :: NFIL
  CHARACTER(*),INTENT(IN) :: NAME,str
  if(len(trim(str)).le.15) then
    WRITE(NFIL,FMT='(55("."),": ",T1,A,T58,A15)')NAME,TRIM(STR)
  else
    WRITE(NFIL,FMT='(55("."),": ",T1,A,T58,A)')NAME,TRIM(STR)
  end if
END SUBROUTINE REPORT_STRING

! ......................................................................
SUBROUTINE REPORT_Rci(nFIL,NAME,VALUE,ci,UNIT)
  INTEGER     ,INTENT(IN) :: NFIL
  CHARACTER(*),INTENT(IN) :: NAME
  REAL(8)     ,INTENT(IN) :: VALUE,ci
  CHARACTER(*),INTENT(IN) :: UNIT
!  IF(DABS(VALUE).LT.1.D+3.AND.DABS(VALUE).GT.1.D-3) THEN
     WRITE(NFIL,FMT='(55("."),": ",T1,A,T58,F15.10," +/-",f15.10," ",A)') &
          NAME,VALUE,ci,UNIT
!!$  ELSE
!!$     WRITE(NFIL,FMT='(55("."),": ",T1,A,T58,ES10.2," +/-",ES10.2," ",A)') &
!!$          NAME,VALUE,ci,UNIT
!!$  END IF
END SUBROUTINE REPORT_RCI
!
!.......................................................................
SUBROUTINE READARGS(VERBOSE,TEMPERATURE,DIR,XMIN,XMAX,NBIN,EUNIT, &
    STARTSKIP,SAMPLESPERWINDOW,TUI,TWHAM,WINDATA,TDAIDXI,&
    segmentwidth,extrrange)
!***********************************************************************
  IMPLICIT NONE
  INTEGER     ,INTENT(INOUT) :: VERBOSE,NBIN
  REAL(8)     ,INTENT(INOUT) :: TEMPERATURE,XMIN,XMAX
  CHARACTER(*),INTENT(INOUT) :: DIR,EUNIT
  INTEGER     ,INTENT(INOUT) :: STARTSKIP,SAMPLESPERWINDOW,WINDATA
  LOGICAL(4)  ,INTENT(INOUT) :: TUI,TWHAM,TDAIDXI
  INTEGER     ,INTENT(INOUT) :: segmentwidth
  real(8)     ,INTENT(INOUT) :: extrrange
  INTEGER                    :: IARG,NUM_ARGS,IARGC,IVAR,IOS
  REAL(8)                    :: SVAR
  CHARACTER(256)             :: LINE
! **********************************************************************
  NUM_ARGS = COMMAND_ARGUMENT_COUNT()
  IARG = 1
  IF(NUM_ARGS.EQ.0) CALL USAGE(1)
  DO
    IF(IARG.GT.NUM_ARGS) EXIT
    CALL GET_COMMAND_ARGUMENT(IARG, LINE)
    CALL UPCASE(LINE)
    IF(LINE(1:1).EQ.'-') THEN
      IF(LINE(2:3).EQ.'H ') THEN
        CALL USAGE(1)
      ELSE IF(TRIM(LINE).EQ.'-UI') THEN
        TWHAM=.FALSE.
      ELSE IF(TRIM(LINE).EQ.'-WHAM') THEN
        TUI=.FALSE.
      ELSE IF(TRIM(LINE).EQ.'-DAIDXI') THEN
        TDAIDXI=.TRUE.
      ELSE IF(TRIM(LINE).EQ.'-T') THEN
        IARG=IARG+1
        CALL GETARG(IARG,LINE)
        READ(LINE,FMT=*,IOSTAT=IOS) SVAR
        IF(IOS.EQ.0) THEN
          TEMPERATURE=SVAR
        ELSE
          PRINT*,'WARNING: USE A REAL NUMBER AS TEMPERATURE: -T 300.'
        END IF
      ELSE IF(TRIM(LINE).EQ.'-D') THEN
        IARG=IARG+1
        CALL GETARG(IARG,LINE)
        READ(LINE,FMT='(a)',IOSTAT=IOS) DIR
        IF(IOS.NE.0) THEN
          PRINT*,'WARNING: ERROR READING DIRECTORY, DEFAULT USED.'
        END IF
      ELSE IF(TRIM(LINE).EQ.'-MIN') THEN
        IARG=IARG+1
        CALL GETARG(IARG,LINE)
        READ(LINE,FMT=*,IOSTAT=IOS) SVAR
        IF(IOS.EQ.0) THEN
          XMIN=SVAR
        ELSE
          PRINT*,'WARNING: USE A REAL NUMBER AS MINIMUM OF THE REACTION COORDINATE: -MIN -5.'
        END IF
      ELSE IF(TRIM(LINE).EQ.'-MAX') THEN
        IARG=IARG+1
        CALL GETARG(IARG,LINE)
        READ(LINE,FMT=*,IOSTAT=IOS) SVAR
        IF(IOS.EQ.0) THEN
          XMAX=SVAR
        ELSE
          PRINT*,'WARNING: USE A REAL NUMBER AS MAXIMUM OF THE REACTION COORDINATE: -MAX 5.'
        END IF
      ELSE IF(TRIM(LINE).EQ.'-N') THEN
        IARG=IARG+1
        CALL GETARG(IARG,LINE)
        READ(LINE,FMT=*,IOSTAT=IOS) IVAR
        IF(IOS.EQ.0) THEN
          NBIN=IVAR
        ELSE
          PRINT*,'WARNING: USE AN INTEGER NUMBER FOR THE NUMBER OF BINS: -N 200'
        END IF
      ELSE IF(TRIM(LINE).EQ.'-SEG') THEN
        IARG=IARG+1
        CALL GETARG(IARG,LINE)
        READ(LINE,FMT=*,IOSTAT=IOS) IVAR
        IF(IOS.EQ.0) THEN
          segmentwidth=IVAR
        ELSE
          PRINT*,'WARNING: USE AN INTEGER NUMBER FOR THE width of each segment in coarse-graining: -SEG 200'
        END IF
      ELSE IF(TRIM(LINE).EQ.'-U') THEN
        IARG=IARG+1
        CALL GETARG(IARG,LINE)
        READ(LINE,FMT=*,IOSTAT=IOS) EUNIT
        IF(IOS.NE.0) THEN
          PRINT*,'WARNING: ERROR READING ENERGY UNIT, DEFAULT USED.'
        END IF
      ELSE IF(TRIM(LINE).EQ.'-SS') THEN
        IARG=IARG+1
        CALL GETARG(IARG,LINE)
        READ(LINE,FMT=*,IOSTAT=IOS) IVAR
        IF(IOS.EQ.0) THEN
          STARTSKIP=IVAR
        ELSE
          PRINT*,'WARNING: USE AN INTEGER NUMBER AS START-SKIP: -SS 1000'
        END IF
      ELSE IF(TRIM(LINE).EQ.'-SPW') THEN
        IARG=IARG+1
        CALL GETARG(IARG,LINE)
        READ(LINE,FMT=*,IOSTAT=IOS) IVAR
        IF(IOS.EQ.0) THEN
          SAMPLESPERWINDOW=IVAR
        ELSE
          PRINT*,'WARNING: USE AN INTEGER NUMBER AS THE NUMBER OF SAMPLES PER WINDOW: -SPW 5000'
        END IF
      ELSE IF(TRIM(LINE).EQ.'-WIN') THEN
        IARG=IARG+1
        CALL GETARG(IARG,LINE)
        READ(LINE,FMT=*,IOSTAT=IOS) IVAR
        IF(IOS.EQ.0) THEN
          WINDATA=IVAR
        ELSE
          PRINT*,'WARNING: USE AN INTEGER NUMBER FOR NUMBER OF WINDOW FOR CONTRIBUTIONS: -WIN 3'
        END IF
      ELSE IF(TRIM(LINE).EQ.'-R') THEN
        IARG=IARG+1
        CALL GETARG(IARG,LINE)
        READ(LINE,FMT=*,IOSTAT=IOS) SVAR
        IF(IOS.EQ.0) THEN
          extrrange=SVAR
        ELSE
          PRINT*,'WARNING: USE A REAL NUMBER AS RANGE FOR COLLECTING EXTREMA: -R 0.4'
        END IF
      ELSE IF(LINE(2:2).EQ.'V') THEN
        IF(TRIM(LINE(3:3)).NE.'') THEN
          READ(LINE(3:),FMT=*,IOSTAT=IOS) IVAR
        ELSE
          IARG=IARG+1
          CALL GETARG(IARG,LINE)
          READ(LINE,FMT=*,IOSTAT=IOS) IVAR
        END IF
        IF(IOS.EQ.0) THEN
          VERBOSE=IVAR
        ELSE
          IARG=IARG-1
          VERBOSE=2
        END IF
      ELSE
        PRINT*,'UNKNOWN ARGUMENT ',TRIM(LINE),' IGNORED'
        CALL USAGE(0)
      END IF
    ELSE
      ! MAYBE INPUT IN OLD STYLE ??
      PRINT*,'UNKNOWN ARGUMENT ',TRIM(LINE),' IGNORED'
      CALL USAGE(0)
    END IF
    IARG=IARG+1
  END DO
END SUBROUTINE READARGS
!
!.......................................................................
SUBROUTINE USAGE(VERBOSE)
! **********************************************************************
  IMPLICIT NONE
  INTEGER   , INTENT(IN)    :: VERBOSE
! **********************************************************************
  IF(VERBOSE.GE.1) THEN
     WRITE(*,*) 'umbrella_integration.x [-option [value]] '
     WRITE(*,*) '    Written in 2005 by Johannes Kaestner. '
     WRITE(*,*) '    kaestner@theochem.uni-stuttgart.de'
     WRITE(*,*)
     WRITE(*,*) ' PURPOSE:'
     WRITE(*,*) '   *) Reads the time dependent reaction coordinate of umbrella sampling'
     WRITE(*,*) '      simulations.'
     WRITE(*,*) '   *) Calculates the free energy in (up to) two ways:'
     WRITE(*,*) '      -) Umbrella integration'
     WRITE(*,*) '      -) Weighed histogram analysis method (WHAM)'
     WRITE(*,*)
     WRITE(*,*) ' INPUT FILE FORMAT:'
     WRITE(*,*) '      A directory with exclusively input files should be provided.'
     WRITE(*,*) '      The data of the windows have to be provided in separate files.'
     WRITE(*,'(a)') '       File format: one line per MD step, "RC-Value force-constant RC-reference-value"'
     WRITE(*,*) '      Where:'
     WRITE(*,*) '        RC-Value is the reaction coordinate of this step,'
     WRITE(*,'(a)') '         force-constant is K of the bias: w(RC) = K/2 * ( RC - RC-reference-value)**2.'
     WRITE(*,*) '        RC-Value and RC-reference-value have to be in the same unit (RCU),'
     WRITE(*,*) '        force-constant has to be in the unit energy_unit * RCU**-2 with'
     WRITE(*,*) '        energy_unit specified with -u.'
     WRITE(*,*) '        force-constant  and RC-reference-value are constant in each file.'
     WRITE(*,*)
  end if
  WRITE(*,*) ' OPTIONS: [default value]'
  WRITE(*,*) '  general options'
  WRITE(*,*) '     -ui      Perform umbrella integration only [ UI & WHAM ]'
  WRITE(*,*) '     -wham    Perform WHAM only [ UI & WHAM ]'
  WRITE(*,*) '     -d dir   Directory containing the data files [ data ]'
  WRITE(*,*) '     -T val   Temperature in K [ 300 ]'
  WRITE(*,*) '     -min val Minimum of the reaction coordinate (analysis starts at that'
  WRITE(*,*) '              value) [ -2. ]'
  WRITE(*,*) '     -max val Maximum of the reaction coordinate (analysis ends at that'
  WRITE(*,*) '              value) [ 2. ]'
  WRITE(*,*) '     -n val   Number of bins [ 200 ]'
  WRITE(*,*) '     -u str   Energy unit of the bias: "au" for atomic units, or "kcal" for '
  WRITE(*,*) '              kcal/mol, or "kj" for kJ/mol [ au ]'
  WRITE(*,*) '     -ss val  Number of MD steps in the data files to skip at start [ 0 ]'
  WRITE(*,*) '     -spw val Maximum number of MD steps to be used per window, use the last'
  WRITE(*,*) '              val steps [ use all ]'
  WRITE(*,*) '  Umbrella Integration specific options'
  WRITE(*,*) '     -win val Plot contributions of window val [ none ]'
  WRITE(*,*) '     -daidxi  Plot truncated mean force of each window (files daidxi.xy and '
  WRITE(*,*) '              daidxi_ximax.xy)'
  WRITE(*,*) '     -seg val Segment width for coarse graining in the error bar estimation'
  WRITE(*,*) '     -r val   Range of the reaction coordinate for collecting extrema. This'
  WRITE(*,*) '              avoids noise to be interpreted as energy extrema. Use a '
  WRITE(*,*) '              negative value to use all extrema.'
  WRITE(*,*) '  output options'
  WRITE(*,*) '     -v 0     Short output (only final energy)'
  WRITE(*,*) '     -v 1     Normal output'
  WRITE(*,*) '     -v 2     Extended output (same as only -v)'  
  WRITE(*,*) '     -v 3     Even more extended output (additional calculations)'  
  WRITE(*,*) '     -h       This help '
  CALL EXIT(0)
END SUBROUTINE USAGE

