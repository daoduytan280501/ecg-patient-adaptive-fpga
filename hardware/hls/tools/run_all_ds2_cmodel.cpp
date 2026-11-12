#include "ecg_f2a.h"
#include "weights_f2a.h"
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){
  int N=atoi(argv[1]);
  FILE*fb=fopen("beats.bin","rb"),*fr=fopen("rr.bin","rb"),*fe=fopen("emb.bin","wb"),*fl=fopen("logits.bin","wb");
  static wo_t hw[N_HEAD_W]; static logit_t hb[NUM_CLASSES];
  for(int o=0;o<NUM_CLASSES;o++){hb[o]=DO_BTOT[o];for(int i=0;i<D1_OUT;i++)hw[o*D1_OUT+i]=DO_W[o][i];}
  signed char b[IN_LEN],r[N_RR];
  for(int n=0;n<N;n++){
    fread(b,1,IN_LEN,fb); fread(r,1,N_RR,fr);
    in_t x[IN_LEN]; rr_t rr[N_RR];
    for(int i=0;i<IN_LEN;i++) x[i].range()=b[i];
    for(int i=0;i<N_RR;i++) rr[i].range()=r[i];
    logit_t lg[NUM_CLASSES]; ad1_t em[D1_OUT]; label_t lab;
    ecg_f2a_top(x,rr,hw,hb,lg,em,lab);
    signed char e8[D1_OUT]; int l32[NUM_CLASSES];
    for(int i=0;i<D1_OUT;i++) e8[i]=(signed char)em[i].range().to_int();
    for(int i=0;i<NUM_CLASSES;i++) l32[i]=lg[i].range().to_int();
    fwrite(e8,1,D1_OUT,fe); fwrite(l32,4,NUM_CLASSES,fl);
  }
  fclose(fe);fclose(fl);return 0;}
