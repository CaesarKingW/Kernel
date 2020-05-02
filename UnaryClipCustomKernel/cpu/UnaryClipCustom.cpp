// includes, system
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdint.h>
#include <ctime>

using namespace std;

const int ARRAY_SIZE = 5120000;
typedef int32_t int32;

clock_t totalStart, totalEnd;
clock_t kernelStart, kernelEnd;
int main(int argc, char* argv[]){
    // 计时开始
    totalStart = clock();
    float *in0 = new float[ARRAY_SIZE];
    float *out = new float[ARRAY_SIZE];
    float *in1 = new float[1];
    float *in2 = new float[1];
    ifstream infile;
    ofstream outfile;
    infile.open("in0.txt",ios::in);
    if(!infile.is_open())
        cout<<"Opening data file fails"<<endl;
    else
        cout<<"Opening data file successes"<<endl;
    for(int i=0; i<ARRAY_SIZE; i++){
        infile>>in0[i];
    }
    in1[0] = 20.0;
    in2[0] = 90.0;

    for(int index=0; index<ARRAY_SIZE; index++){
        float value = in2[0] < in0[index] ? in2[0] : in0[index];
        out[index] = value < in1[0] ? in1[0] : value;
    }


    // 检查执行结果
    /*
    for(int i=0; i<ARRAY_SIZE; i++){
        cout<<out[i]<<endl;
    }
    */
    // 计时结束
    totalEnd = clock();
    double totalTime=(double)(totalEnd-totalStart)/CLOCKS_PER_SEC;
	cout<<"Total time:"<<totalTime*1000<<"ms"<<endl;	//ms为单位
    return 0;
}