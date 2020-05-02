#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
using namespace std;

const int ARRAY_SIZE = 1000000;
int main(){
    srand((unsigned)time(NULL));
    double arr[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; i++){
        arr[i] = rand()%100;
    }
    int num = rand() % ARRAY_SIZE;
    arr[num] = 1.0/0.0;
    for(int i = 0; i < ARRAY_SIZE; i++){
        cout<<arr[i]<<endl;
    }
    return 0;
}