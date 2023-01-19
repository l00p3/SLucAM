echo "BUILDING G2O LIBRARY ..."

cd external/g2o
mkdir build
cd build
cmake ../
make -j

echo "G2O LIBRARY BUILT"
