#! /bin/bash
# test/testPrintBug.sh.  Generated from testPrintBug.sh.in by configure.

./testPrintBug 

OS=`uname`
case "$OS" in
CYGWIN*)
  cmd1=`sed 's/e-0\([0-9][0-9]\)/e-\1/g' testPrintBug.out | \
	sed 's/e+0\([0-9][0-9]\)/e+\1/g'  | \
	diff -q -b - ../../hepmc2.06.09/test/testPrintBug.output`
  ;;
Darwin*)
  # MacOSX fix
  cmd1=`sed 's/e-0\([0-9][0-9]\)/e-\1/g' testPrintBug.out | \
	sed 's/e+0\([0-9][0-9]\)/e+\1/g'  | \
	diff -q -b - ../../hepmc2.06.09/test/testPrintBug.output`
  ;;
*)
  cmd1=`diff -q -b testPrintBug.out ../../hepmc2.06.09/test/testPrintBug.output`
esac

if [ -n "$cmd1" ]
then
  echo $cmd1
  exit 1;
fi

exit 0;
