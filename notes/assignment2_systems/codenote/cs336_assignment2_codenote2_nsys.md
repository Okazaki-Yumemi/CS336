前面配置环境就不多说了


```bash

cd ~/projects/CS336-2026/assignments/a2-systems

mkdir -p profiles

uv run nsys profile \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  --output=profiles/small_forward_test \
  -- python scripts/benchmark.py \
  --model-size small \
  --context-length 128 \
  --mode forward \
  --warmup-steps 1 \
  --measurement-steps 1
```


```bash

(cs336-systems) (base) soyo@localhost:~/projects/CS336-2026/assignments/a2-systems$ nsys stats \
  --report cuda_gpu_kern_sum \
  --report cuda_api_sum \
  profiles/small_forward_test.nsys-rep
Generating SQLite file profiles/small_forward_test.sqlite from profiles/small_forward_test.nsys-rep
Processing [profiles/small_forward_test.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     41.6         12009038         74  162284.3  209581.5     12240    699104     132526.4  void magma_sgemmEx_kernel<float, float, float, (bool)1, (bool)0, (int)6, (int)4, (int)6, (int)3, (i…
     27.4          7901612         96   82308.5   80266.5     78251    267989      19217.9  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn_align1>(T1::Params)                       
     18.0          5190917         24  216288.2  207665.5    204340    404158      40269.0  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(T1::Params)                      
      2.5           719215        292    2463.1    2189.0      2016      6279        483.8  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      1.3           379473         48    7905.7    7401.5      6796     18289       1875.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, floa…
      1.2           352662        144    2449.0    1411.0      1324      5184       1508.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::arr…
      1.1           303067         48    6313.9    6164.0      6048     12269        880.9  void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>,…
      1.0           280029         24   11667.9   11535.0     11289     12471        353.1  void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_align1>(T1::Params)                        
      0.8           228792         24    9533.0    3211.0      2995    154314      30838.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::exp_kernel_cuda(at::TensorIterat…
      0.6           178452         24    7435.5    7459.0      6769      7776        212.8  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MaxOps<flo…
      0.6           168969         24    7040.4    6768.0      6624     12298       1130.8  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      0.6           159611         48    3325.2    3283.0      3053      5818        393.6  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      0.5           157392         24    6558.0    6537.0      6077      7114        214.3  void at::native::vectorized_elementwise_kernel<(int)4, at::native::sigmoid_kernel_cuda(at::TensorIt…
      0.5           136602         24    5691.8    5443.5      5242     10426       1030.4  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      0.4           122745         50    2454.9    2419.0      1959      3283        266.0  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<fl…
      0.4           117476         24    4894.8    4954.0      4262      5702        342.4  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapp…
      0.4           116609         24    4858.7    4838.0      4666      5126        130.0  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      0.3            98208         50    1964.2    1929.0      1843      2650        146.7  void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scala…
      0.3            78826         24    3284.4    3168.0      3082      5415        471.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<float, float, floa…
      0.2            45794         50     915.9     778.0       748      1584        201.4  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, st…
      0.1            38769         50     775.4     777.5       720       892         36.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::rsqrt_kernel_cuda(at::TensorIter…
      0.1            29898         24    1245.8    1210.0      1180      1728        113.7  void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n…
      0.1            16764         24     698.5     604.5       490      1066        189.4  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<c…
      0.0            13132          2    6566.0    6566.0      6307      6825        366.3  void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, lon…
      0.0            11833         24     493.0     489.5       432       605         31.0  void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &,…
      0.0             1815          2     907.5     907.5       749      1066        224.2  void at::native::<unnamed>::distribution_elementwise_grid_stride_kernel<unsigned int, (int)4, void …

Processing [profiles/small_forward_test.sqlite] with [/opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                Name              
 --------  ---------------  ---------  ----------  ---------  --------  --------  -----------  --------------------------------
     56.7        266228432         21  12677544.4  8693526.0    351974  36203572   10204361.7  cuLibraryLoadData               
     23.2        109080908       1122     97220.1     5681.0      2362  13211976     822632.8  cudaLaunchKernel                
     10.6         49966736        112    446131.6   251014.0      4067   3763545     516786.7  cudaMemcpyAsync                 
      6.0         28027214         64    437925.2   407347.0      7385   1341184     173461.0  cudaMalloc                      
      1.3          6195770        112     55319.4    53011.0     12867    140235      30558.3  cudaStreamSynchronize           
      0.4          1915046         28     68394.5     1767.0       622    579458     158846.9  cuLibraryGetKernel              
      0.4          1703529        120     14196.1     9607.5      3489    332019      30268.5  cudaMemsetAsync                 
      0.3          1556444        144     10808.6     6524.0      2921     78684      11367.9  cuLaunchKernel                  
      0.3          1417348        177      8007.6     1028.0       163   1185237      89008.4  cudaStreamIsCapturing           
      0.3          1226535          3    408845.0    11145.0      2840   1212550     696041.3  cudaDeviceSynchronize           
      0.2           754671        144      5240.8      563.5       135    476996      41200.5  cuKernelGetFunction             
      0.2           712975       1122       635.5      154.0        72    404454      12088.8  cuKernelGetName                 
      0.1           374494         18     20805.2      239.0       153    368462      86764.4  cudaEventCreateWithFlags        
      0.0           161741        880       183.8      126.0        54     12678        442.1  cuGetProcAddress_v2             
      0.0            27494          1     27494.0    27494.0     27494     27494          0.0  cudaFree                        
      0.0             9496          2      4748.0     4748.0      4464      5032        401.6  cudaGetDeviceProperties         
      0.0             7080          4      1770.0     1808.0      1146      2318        565.4  cuInit                          
      0.0             5897          3      1965.7      306.0       153      5438       3008.1  cuModuleGetLoadingMode          
      0.0             1778          3       592.7      469.0       154      1155        511.8  cudaGetDriverEntryPointByVersion
```

# 1. GPU时间主要用于矩阵乘法

41.6%  magma_sgemmEx
27.4%  CUTLASS GEMM
18.0%  CUTLASS GEMM

合计 87%

# 2. 当前报告混入了一次性初始化

cuLibraryLoadData    266 ms
cudaMalloc            28 ms

# 3. 大量kernel launch

cudaLaunchKernel: 1122 calls

代码修改

```py
    with torch.cuda.nvtx.range("measurement"):
        for _ in range(measurement_steps):
            torch.cuda.synchronize()
            
            start = timeit.default_timer()
            
            run_step(
                model,
                inputs,
                targets,
                mode,
                optimizer
            )
            torch.cuda.synchronize()
            end = timeit.default_timer()
            
            times.append(end-start)
```
用 torch.cuda.nvtx.range("measurement") 包裹measurement函数

