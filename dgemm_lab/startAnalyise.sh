/opt/oracle/solstudiodev/lib/analyzer/lib/../../../bin/collect -o test.5.er -p on -h PAPI_fp_ops,on -S on -A on /xbar/nas1/home1/s06/s062407/HPC/mm

er_print -lines test.5.er/ |grep simple_mm | awk {'print $4'};
