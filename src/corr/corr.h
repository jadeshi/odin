
class Corr
{
  int N;
  short mean_norm;
  float ar1_mean, ar2_mean;
  float ar1_stdev, ar2_stdev;
  
  float mean_no_zero(float * ar);
  float stdev_no_zero(float * ar, float ar_mean);
  
  void correlate_mean_norm(float * ar1, float * ar2, float * ar3);
  void correlate_stdev_norm(float * ar1, float * ar2, float * ar3);

public:
  Corr(int N_, float * ar1, float * ar2, float * ar3, short mean_norm_);
  ~Corr();
};

