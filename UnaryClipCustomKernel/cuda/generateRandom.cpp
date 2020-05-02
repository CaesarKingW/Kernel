#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
using namespace std;

const int ARRAY_SIZE = 5120000;
int main(){
    srand((unsigned)time(NULL));
    double *arr = new double[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; i++){
        arr[i] = rand()%200;
    }
    for(int i = 0; i < ARRAY_SIZE; i++){
        cout<<arr[i]<<endl;
    }
    return 0;
}