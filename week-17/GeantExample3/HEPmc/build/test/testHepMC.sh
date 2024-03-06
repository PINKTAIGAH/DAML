#! /bin/bash
# test/testHepMC.sh.  Generated from testHepMC.sh.in by configure.

# cleanup any old output files first
rm -f testHepMC.cout
rm -f testHepMC.out
rm -f testHepMCParticle.out
rm -f testIOAscii.dat
rm -f testCrossSection.cout
rm -f testCrossSection.out
rm -f testCrossSection2.out
rm -f testWithWeight.cout
rm -f testWithWeight.out
rm -f testWithWeight2.out

./testHepMC

OS=`uname`
case "$OS" in
CYGWIN*)
  cmd1=`sed 's/e-0/e-/g' testHepMC.out | \
        sed 's/e+0/e+/g' | \
        sed 's/MEV/GEV/g' | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
  cmd3=`sed 's/e-0/e-/g' testCrossSection.out | \
        sed 's/e+0/e+/g' | \
        sed 's/MEV/GEV/g' | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testCrossSection.dat`
  cmd5=`sed 's/e-0/e-/g' testCrossSection.cout | \
        sed 's/e+0/e+/g' | \
        sed 's/MEV/GEV/g' | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testCrossSection.output`
  cmd6=`sed 's/e-0/e-/g' testWithWeight.out | \
        sed 's/e+0/e+/g' | \
        sed 's/MEV/GEV/g' | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testWithWeight.dat`
  cmd8=`sed 's/e-0/e-/g' testWithWeight.cout | \
        sed 's/e+0/e+/g' | \
        sed 's/MEV/GEV/g' | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testWithWeight.output`
if [ "MEV" = "GEV" ];then
  if [ "MM" = "MM" ];then
  cmd9=`sed 's/e-0/e-/g' testHepMCVarious.cout | \
        sed 's/e+0/e+/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMCVarious.output`
  fi
fi
  cmd=`sed 's/e-0/e-/g' testHepMC.cout | \
        sed 's/e+0/e+/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMC.output`
  #don't compare these on Windows (hopelessly different default output)
  cmd2=
  cmd4=`diff -q -b testCrossSection2.out testCrossSection.out`
  cmd7=`diff -q -b testWithWeight2.out testWithWeight.out`
  cmd10=`diff -q -b testDefineUnits.out testHepMC.out`
  ;;
Darwin*)
  # MacOSX fix
  cmd1=`sed 's/MEV/GEV/g' testHepMC.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
  cmd3=`sed 's/MEV/GEV/g' testCrossSection.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testCrossSection.dat`
  cmd5=`sed 's/e-00/e+00/g' testCrossSection.cout | \
        sed 's/MEV/GEV/g' | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testCrossSection.output`
  cmd6=`sed 's/MEV/GEV/g' testWithWeight.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testWithWeight.dat`
  cmd8=`sed 's/MEV/GEV/g' testWithWeight.cout | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testWithWeight.output`
if [ "MEV" = "GEV" ];then
  if [ "MM" = "MM" ];then
    cmd9=`diff -q -b testHepMCVarious.cout ../../hepmc2.06.09/test/testHepMCVarious.output`
  fi
fi
  cmd=`diff -q -b testHepMC.cout ../../hepmc2.06.09/test/testHepMC.output`
  cmd2=`sed 's/e-00/e+00/g'  ../../hepmc2.06.09/test/testHepMCParticle.output | \
	diff -q -b - testHepMCParticle.out`
  cmd4=`diff -q -b testCrossSection2.out testCrossSection.out`
  cmd7=`diff -q -b testWithWeight2.out testWithWeight.out`
  cmd10=`diff -q -b testDefineUnits.out testHepMC.out`
  ;;
*)

  cmd1=`sed 's/MEV/GEV/g' testHepMC.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
  cmd3=`sed 's/MEV/GEV/g' testCrossSection.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testCrossSection.dat`
  cmd5=`sed 's/e-00/e+00/g' testCrossSection.cout | \
        sed 's/MEV/GEV/g' | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testCrossSection.output`
  cmd6=`sed 's/MEV/GEV/g' testWithWeight.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testWithWeight.dat`
  cmd8=`sed 's/e-00/e+00/g' testWithWeight.cout | \
        sed 's/MEV/GEV/g' | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testWithWeight.output`
if [ "MEV" = "GEV" ];then
  if [ "MM" = "MM" ];then
    cmd9=`diff -q -b testHepMCVarious.cout ../../hepmc2.06.09/test/testHepMCVarious.output`
  fi
fi
  cmd=`diff -q -b testHepMC.cout ../../hepmc2.06.09/test/testHepMC.output`
  cmd2=`diff -q -b testHepMCParticle.out ../../hepmc2.06.09/test/testHepMCParticle.output`
  cmd4=`diff -q -b testCrossSection2.out testCrossSection.out`
  cmd7=`diff -q -b testWithWeight2.out testWithWeight.out`
  cmd10=`diff -q -b testDefineUnits.out testHepMC.out`
esac

if [ -n "$cmd" ]; then
  echo "testHepMC.cout and ../../hepmc2.06.09/test/testHepMC.output differ"
  exit 1;
fi

if [ -n "$cmd1" ]
then
  echo $cmd1
  exit 1;
fi

if [ -n "$cmd2" ]
then
  echo $cmd2
  exit 1;
fi

if [ -n "$cmd3" ]
then
  echo $cmd3
  exit 1;
fi

if [ -n "$cmd4" ]
then
  echo $cmd4
  exit 1;
fi

if [ -n "$cmd5" ]
then
  echo $cmd5
  exit 1;
fi

if [ -n "$cmd6" ]
then
  echo $cmd6
  exit 1;
fi

if [ -n "$cmd7" ]
then
  echo $cmd7
  exit 1;
fi

if [ -n "$cmd8" ]
then
  echo $cmd8
  exit 1;
fi

if [ -n "$cmd9" ]
then
  echo $cmd9
  exit 1;
fi

if [ -n "$cmd10" ]
then
  echo $cmd10
  exit 1;
fi

exit 0;

