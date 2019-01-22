reset
set terminal pngcairo size 640, 640 #enhanced font "Times New Roman, 10"
set output "outputSerialVsGPGPU.png"



dataDir = "../../Cuda\\ One\\ Ring/experiments/acquired_meshes/"
#dataDir = "../../Cuda One Ring/experiments/synthetic_meshes/"

#dataName = "ILATO"
#data1a = dataDir."ILATO_1A_SM2066-HE5-60_070214_merged_GMO_ASCII_funcvals_"

dataName = "Unisiegel"
data1a = dataDir."Unisiegel_UAH_Ebay-Siegel_Uniarchiv_HE2066-60_010614_partial_ASCII_funcvals_"

data1b = "iter_gigamesh.txt" 
data2a = data1a
data2b = "iter_libcudaonering.txt"



set title "Difference between Function Value Outputs\nfrom Serial vs GPGPU Computation at Increasing Iterations" 

#plot '< paste '.data1.' '.data2 u ($2-$4)
#plot '< tail -n +12 '.data1.' | paste - '.data2 u ($2-$4) with points title columnhead(1)

plot for [i in "1 3 10 30 100 300 1000 3000"] '< tail -n +12 '.data1a.i.data1b.' | paste - '.data2a.i.data2b u ($2-$4) with lines title dataName." ".i." iters"
