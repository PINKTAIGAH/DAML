ENERGY_RANGE=($(seq 10000 100 100000))

for i in ${ENERGY_RANGE[@]}; do
    > ./run.mac
    COMMAND="/control/verbose 2\n/control/saveHistory\n/gun/particle e-\n/gun/energy ${i} MeV\n/run/beamOn 0\n/run/beamOn 1000"
    echo -en "${COMMAND}" >> ./run.mac
    ./build/MyProgram
    mv output_nt_Energy.csv "data/2d_hist_data/electron_${i}mev_1000.csv"
    echo "1000 ${i} MeV electrons simulated"
done

rm *csv *ps
echo "Simulation finished"