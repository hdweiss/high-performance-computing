void sum_gold( float* output, float* DataSet, const unsigned int N) 
{
  output[0] = 0;
  float sum = 0;
  for(unsigned int i = 0; i < N; ++i) 
      sum += DataSet[i];
  output[0] = sum;
}


