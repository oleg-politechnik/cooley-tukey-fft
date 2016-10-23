#define _USE_MATH_DEFINES

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>

#define FFT_SIZE 2*3*5

#define RE(arr, ix) arr[(ix)*2]
#define IM(arr, ix) arr[(ix)*2+1]

float m_sourceData[FFT_SIZE * 2];
float m_directDftData[FFT_SIZE * 2];
float m_directCfftData[FFT_SIZE * 2];
float m_directCfftDataTemp[FFT_SIZE * 2];

#define FFT_STACK_SIZE 30

void fillIn(float data[])
{
    for (int i = 0; i < FFT_SIZE; i++)
    {
      RE(data, i) = sin(1 * 2 * M_PI * i / FFT_SIZE) * 128 + sin(
              3 * 2 * M_PI * i / FFT_SIZE) * 32;
      IM(data, i) = 0;
    }
}

typedef struct
{
  int L_n;
  int M_n;
  int s_n;
  int n_prev;
  float temp[FFT_SIZE * 2];
} fft_stack_frame_t;

static void fill_stack(int num, int *stack_depth, fft_stack_frame_t fft_stack[])
{
  int current_divider = 2;
  int current_num;
  int rest;

  fft_stack[0].L_n = 1;
  fft_stack[0].M_n = FFT_SIZE;
  fft_stack[0].s_n = 0;
  fft_stack[0].n_prev = 0;

  *stack_depth = 1;

  while ((current_divider < (current_num = fft_stack[*stack_depth - 1].M_n))
          & (*stack_depth < FFT_STACK_SIZE))
  {
    rest = current_num / current_divider;
    if ((double) current_num / current_divider == (double) rest)
    {
      fft_stack[*stack_depth].L_n = current_divider;
      fft_stack[*stack_depth].M_n = rest;

      (*stack_depth)++;
    }
    else
    {
      current_divider++;
    }
  }
}

void print_array(float array[])
{
  for (int i = 0; i < FFT_SIZE; ++i)
  {
    printf("%d: (%6.1f, %6.1f)\n", i, RE(array, i), IM(array, i));
  }
}

static inline void WmulAc(double *im_acc, double *re_acc, double im_x,
        double re_x, unsigned int power, unsigned int base)
{
  double w, wr, wi;
  w = 2.0 * M_PI * power / base;
  wr = cos(w);
  wi = sin(w);
  (*im_acc) += im_x * wr + re_x * wi;
  (*re_acc) += re_x * wr - im_x * wi;
}

void ComputeDft(unsigned int length, float dataIn[], float dataOut[])
{
  double tempIm, tempRe;

  for (unsigned int k = 0; k < length; k++)
  {
    tempIm = tempRe = 0;
    for (unsigned int n = 0; n < length; n++)
    {
      WmulAc(&tempIm, &tempRe, IM(dataIn, n), RE(dataIn, n), n * k, length);
    }

    RE(dataOut, k) = tempRe;
    IM(dataOut, k) = tempIm;
  }
}

void ComputeCfft(int stage, int max_stage, fft_stack_frame_t fft_stack[],
        float dataOut[])
{
  double tempIm, tempRe;
  int n, n_out, k;

  int M_n = fft_stack[stage].M_n;
  int L_n = fft_stack[stage].L_n;

  int L_prev, M_prev, n_prev;

  L_prev = fft_stack[stage - 1].L_n;
  M_prev = fft_stack[stage - 1].M_n;
  n_prev = fft_stack[stage - 1].n_prev;

  //L-point DFT of columns and rotating coefficients
  for (int m = 0; m < M_n; m++)
  {
    for (int s = 0; s < L_n; s++)
    {
      n_out = n_prev + M_n * s + m;

      tempIm = tempRe = 0;

      for (int l = 0; l < L_n; l++)
      {
        n = n_prev + M_n * l + m;
        WmulAc(&tempIm, &tempRe, IM(fft_stack[stage - 1].temp, n),
                RE(fft_stack[stage - 1].temp, n), M_n * s * l + m * s,
                M_n * L_n);
      }

      IM(fft_stack[stage].temp, n_out) = tempIm;
      RE(fft_stack[stage].temp, n_out) = tempRe;
    }
  }

  int k_add, k_mul;

  if (stage == max_stage)
  {
    k_add = fft_stack[0].s_n;
    k_mul = 1;

    for (int i = 1; i < stage; ++i) {
      k_mul *= fft_stack[i-1].L_n;
      k_add += fft_stack[i].s_n * k_mul;
    }

    k_mul *= fft_stack[stage-1].L_n;
  }

  //M-point DFT of arrows -> split
  for (int s = 0; s < L_n; s++)
  {
    if (stage < max_stage)
    {
      fft_stack[stage].s_n = s;
      fft_stack[stage].n_prev = fft_stack[stage - 1].n_prev + M_n * s;
      ComputeCfft(stage + 1, max_stage, fft_stack, dataOut);
    }
    else
    {
      for (int r = 0; r < M_n; r++)
      {
        //1 поток (m,s)

        k = k_add + k_mul * (L_n * r + s); // Транспонирование
        tempIm = tempRe = 0;

        for (int m = 0; m < M_n; m++)
        {
          n = n_prev + M_n * s + m;
          WmulAc(&tempIm, &tempRe, IM(fft_stack[stage].temp, n),
                  RE(fft_stack[stage].temp, n), m * r, M_n);
        }

        IM(dataOut, k) = tempIm;
        RE(dataOut, k) = tempRe;
      }
    }
  }
}

//-----------------------------------------------------------------

int main(int argc, char *argv[])
{

  fft_stack_frame_t fft_stack[FFT_STACK_SIZE];
  int stack_depth;

  //assert_param stack_depth > 1

  fill_stack(FFT_SIZE, &stack_depth, fft_stack);

  printf("CTFT call stack:\n");
  for (int i = 0; i < stack_depth; ++i)
  {
    printf("Frame %d: (%d x %d)\n", i, fft_stack[i].L_n, fft_stack[i].M_n);
  }
  printf("\n");

  // Compute regular DFT to check
  fillIn(m_sourceData);
  print_array(m_sourceData);
  ComputeDft(FFT_SIZE, m_sourceData, m_directDftData);

  fillIn(fft_stack[0].temp);
  ComputeCfft(1, stack_depth - 1, fft_stack, m_directCfftData);

  //
  bool error = false;
  bool not_null = false;

  for (int i = 0; i < FFT_SIZE; ++i)
  {
    if ((int) roundf(RE(m_directDftData, i)) != (int) roundf(
            RE(m_directCfftData, i)) || (int) roundf(
            IM(m_directDftData, i)) != (int) roundf(
            IM(m_directCfftData, i)))
      error = true;

    if (RE(m_directDftData, i) != 0 || IM(m_directDftData, i) != 0)
      not_null = true;
  }

  if (error)
  {
    for (int i = 0; i < FFT_SIZE; ++i)
    {
      printf("%d: (%d, %d) vs (%d, %d)\n", i,
              (int) roundf(RE(m_directDftData, i)),
              (int) roundf(IM(m_directDftData, i)),
              (int) roundf(RE(m_directCfftData, i)),
              (int) roundf(IM(m_directCfftData, i)));
    }
    printf("ERROR\n");
  }
  else
  {
    if (!not_null)
      printf("NULL\n");
    else
    {
      print_array(m_directCfftData);
      printf("OK\n");
    }
  }

  return 0;
}

