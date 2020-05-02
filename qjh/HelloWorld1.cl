__kernel void hello_kernel(__global const bool *a,
	__global const bool *b,
    __global int *result)
{
    int gid = get_global_id(0);
    int c[8] = {1,1,1,1,1,1,1,1};
    if(a[gid]==b[0])
    	c[0]=0;
    if(a[gid+1]==b[0])
    	c[1]=0;
    if(a[gid+2]==b[0])
    	c[2]=0;
    if(a[gid+3]==b[0])
    	c[3]=0;
    if(a[gid+4]==b[0])
    	c[4]=0;
    if(a[gid+5]==b[0])
    	c[5]=0;
    if(a[gid+6]==b[0])
    	c[6]=0;
    if(a[gid+7]==b[0])
    	c[7]=0;
    result[gid] = c[0]*128+c[1]*64+c[2]*32+c[3]*16+c[4]*8+c[5]*4+c[6]*2+c[7]*1;
       
}