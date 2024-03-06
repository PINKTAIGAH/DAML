#! /bin/bash
# test/testStreamIO.sh.  Generated from testStreamIO.sh.in by configure.

# cleanup any old output files first
rm -f testStreamIO.out
rm -f testStreamIO2.out
rm -f testStreamIOParticle.out
rm -f testStreamIOVarious.out
rm -f testStreamIO.cout

./testStreamIO

OS=`uname`
case "$OS" in
CYGWIN*)
  cmd1=`sed 's/e-0/e-/g' testStreamIO.out | \
        sed 's/e+0/e+/g' | \
        sed 's/MEV/GEV/g' | \
        sed 's/MM/MM/g' | \
         diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
  cmd2=`sed 's/e-0/e-/g' testStreamIO2.out | \
        sed 's/e+0/e+/g' | \
        sed 's/MEV/GEV/g' | \
        sed 's/MM/MM/g' | \
         diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
  cmd5=`sed 's/e-0/e-/g' testStreamIO3.out | \
        sed 's/e+0/e+/g' | \
        sed 's/MEV/GEV/g' | \
        sed 's/MM/MM/g' | \
         diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
  cmd6=`sed 's/e-0/e-/g' testStreamIO4.out | \
        sed 's/e+0/e+/g' | \
        sed 's/MEV/GEV/g' | \
        sed 's/MM/MM/g' | \
         diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
if [ "MEV" = "GEV" ];then
  if [ "MM" = "MM" ];then
    cmd9=`diff -q -b testStreamIOVarious.cout ../../hepmc2.06.09/test/testStreamIOVarious.output`
  fi
fi
  cmd=`diff -q -b testStreamIO.cout ../../hepmc2.06.09/test/testStreamIO.output`
  #don't compare IOParticle on Windows (hopelessly different default output)
  cmd3=
  cmd4=`diff -q -b testStreamIOVarious.out testNaN.out`
  ;;
Darwin*)
  # MacOSX fix
  cmd1=`sed 's/MEV/GEV/g' testStreamIO.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
  cmd2=`sed 's/MEV/GEV/g' testStreamIO2.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
  cmd5=`sed 's/MEV/GEV/g' testStreamIO3.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
  cmd6=`sed 's/MEV/GEV/g' testStreamIO4.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
if [ "MEV" = "GEV" ];then
  if [ "MM" = "MM" ];then
    cmd9=`diff -q -b testStreamIOVarious.cout ../../hepmc2.06.09/test/testStreamIOVarious.output`
  fi
fi
  cmd=`diff -q -b testStreamIO.cout ../../hepmc2.06.09/test/testStreamIO.output`
  cmd3=`sed 's/e-00/e+00/g'  ../../hepmc2.06.09/test/testHepMCParticle.output | \
	diff -q -b - testStreamIOParticle.out`
  cmd4=`diff -q -b testStreamIOVarious.out testNaN.out`
  ;;
*)
  cmd1=`sed 's/MEV/GEV/g' testStreamIO.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
  cmd2=`sed 's/MEV/GEV/g' testStreamIO2.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
  cmd5=`sed 's/MEV/GEV/g' testStreamIO3.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
  cmd6=`sed 's/MEV/GEV/g' testStreamIO4.out | \
        sed 's/MM/MM/g' | \
        diff -q -b - ../../hepmc2.06.09/test/testHepMC.dat`
if [ "MEV" = "GEV" ];then
  if [ "MM" = "MM" ];then
    cmd9=`diff -q -b testStreamIOVarious.cout ../../hepmc2.06.09/test/testStreamIOVarious.output`
  fi
fi
  cmd=`diff -q -b testStreamIO.cout ../../hepmc2.06.09/test/testStreamIO.output`
  cmd3=`diff -q -b testStreamIOParticle.out ../../hepmc2.06.09/test/testHepMCParticle.output`
  cmd4=`diff -q -b testStreamIOVarious.out testNaN.out`
esac

if [ -n "$cmd" ]; then
  echo $cmd
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

if [ -n "$cmd9" ]
then
  echo $cmd9
  exit 1;
fi

exit 0;


