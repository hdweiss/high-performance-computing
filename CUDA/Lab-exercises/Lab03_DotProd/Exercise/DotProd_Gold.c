void DotProd_Gold(float* Vec1, float* Vec2, float *Output, int N)
{
   *Output = 0.0f;
   for (int i = 0; i < N; ++i)
      *Output +=  Vec1[i] * Vec2[i]; 
}