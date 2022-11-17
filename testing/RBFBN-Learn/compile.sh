# g++ -o flann-test -I /usr/include/hdf5/serial/ -O2 flann-test.cpp -L /usr/lib/x86_64-linux-gnu/hdf5/serial/ -L /usr/lib/x86_64-linux-gnu/ -llz4 -lhdf5
g++ -o test -I /usr/include/hdf5/serial/ -I /home/ros/code/flann/src/cpp -ggdb test.cpp -L /usr/lib/x86_64-linux-gnu/hdf5/serial/ -L /home/ros/code/flann/lib -llz4 -lhdf5 -lnlopt
