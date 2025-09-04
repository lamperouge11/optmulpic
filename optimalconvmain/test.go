package main
package main

import (
	"fmt"
	"strconv"
	"time"
	"math"

	"github.com/dwkim606/test_lattigo/ckks"
)

// Fast Conv without boot, Assume full batch with Po2 in_wid & N
// Normal Conv without output modification (e.g., trimming or expanding)
// Assume that the input is 0 padded according to kernel size: only in_wid - (ker_wid-1)/2 elements in row and columns are nonzero
// Also support non-full batching case
func testConv_in(in_batch, in_wid, ker_wid, total_test_num int, boot bool) {
	kind := "Conv_xc"
	printResult := false
	raw_in_batch := in_batch         // same as python
	raw_in_wid := in_wid - ker_wid/2 // same as python
	norm := in_batch / raw_in_batch
	test_dir := "test_conv_data/"
	pow := 4.0

	// set basic variables for above input variables
	kp_wid, out_batch, logN, trans := set_Variables(in_batch, raw_in_wid, in_wid, ker_wid, kind)
	raw_out_batch := out_batch / norm

	
	// generate Context: params, Keys, rotations, general plaintexts
	cont := newContext(logN, ker_wid, []int{in_wid}, []int{kp_wid}, boot, kind)
	fmt.Println("vec size: log2 = ", cont.logN)
	fmt.Println("raw input width: ", raw_in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num raw batches in & out: ", raw_in_batch, ", ", raw_out_batch)


	for test_iter := 0; test_iter < total_test_num; test_iter++ {
		fmt.Println(test_iter+1, "-th iter...start")
		raw_input := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_in_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*raw_in_batch)
		ker_in := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_ker_"+strconv.Itoa(test_iter)+".csv", raw_in_batch*raw_out_batch*ker_wid*ker_wid)
		bn_a := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_bna_"+strconv.Itoa(test_iter)+".csv", raw_out_batch)
		bn_b := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_bnb_"+strconv.Itoa(test_iter)+".csv", raw_out_batch)

		// input encryption
		input := prep_Input(raw_input, raw_in_wid, in_wid, cont.N, norm, trans, printResult)
		start := time.Now()
		plain_tmp := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, plain_tmp)
		ctxt_input := cont.encryptor.EncryptNew(plain_tmp)
		fmt.Printf("Encryption done in %s \n", time.Since(start))
		startevalConv_BN := time.Now()
		// Kernel Prep & Conv (+BN) Evaluation
		var ct_result *ckks.Ciphertext
		if boot {
			ct_result = evalConv_BNRelu_new(cont, ctxt_input, ker_in, bn_a, bn_b, 0.0, pow, in_wid, kp_wid, ker_wid, raw_in_batch, raw_out_batch, norm, 0, 0, 2, 0, kind, false, false)
					
			//ct_result = evalConv_BN(cont, ctxt_input, ker_in, bn_a, bn_b, in_wid, ker_wid, raw_in_batch, raw_out_batch, norm, math.Exp2(math.Round(math.Log2(float64(cont.params.Q()[0]))-(pow+8))), trans)
			// math.Exp2(math.Round(math.Log2(float64(cont.params.Q()[0]))-(pow+8)))
			//ct_result.Scale = ct_result.Scale * math.Pow(2, pow)
			//ct_result = cont.btp.Bootstrapp(ct_result)
			//cont.evaluator.MulByPow2(ct_result, int(pow), ct_result)
			//ct_result = cont.evaluator.MulNew(ct_result,ct_result)
			
			//ct_result.Scale = ct_result.Scale * math.Pow(2, 2)
			
			//cont.evaluator.Rescale(ct_result, cont.params.Scale(), ct_result)
						
			//ct_result = evalReLU(cont.params, cont.evaluator, ct_result, 0.0)
			
			//cont.evaluator.MulByPow2(ct_result, int(pow), ct_result)
			//cont.evaluator.Rescale(ct_result, cont.params.Scale(), ct_result)
			
			
		} else {
			ct_result = evalConv_BN(cont, ctxt_input, ker_in, bn_a, bn_b, in_wid, ker_wid, raw_in_batch, raw_out_batch, norm, float64(1<<30), trans)
		}
		fmt.Printf("evalConv_BN evalReLU Done in %s \n", time.Since(startevalConv_BN))
		start = time.Now()
		cont.decryptor.Decrypt(ct_result, plain_tmp)
		cfs_tmp := cont.encoder.DecodeCoeffs(plain_tmp)
		fmt.Printf("Decryption Done in %s \n", time.Since(start))

		test_out := post_process(cfs_tmp, raw_in_wid, in_wid)
		var real_out []float64
		if boot {
			real_out = readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_reluout_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*raw_in_batch)
		} else {
			real_out = readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_out_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*raw_in_batch)
		}

		printDebugCfsPlain(test_out, real_out)
	}

}

func testConv_in_xuchao(in_batch, in_wid, ker_wid, total_test_num, split int, boot bool) {
	kind := "Conv_xc"
	printResult := false
	raw_in_batch := in_batch         // same as python
	raw_in_wid := in_wid - ker_wid/2 // same as python
	norm := in_batch / raw_in_batch
	test_dir := "test_conv_data/"
	pow := 4.0
	split_pow := split
	split_batch := 1<<split_pow

	// set basic variables for above input variables
	kp_wid, out_batch, logN, trans := set_Variables(in_batch, raw_in_wid, in_wid, ker_wid, kind)
	raw_out_batch := out_batch / norm

	// generate Context: params, Keys, rotations, general plaintexts
	cont := newContext(logN - split_pow, ker_wid, []int{in_wid}, []int{kp_wid}, boot, kind)
	
	fmt.Println("vec size: log2 = ", cont.logN)
	fmt.Println("raw input width: ", raw_in_wid)
	fmt.Println("kernel width: ", ker_wid)
	fmt.Println("num raw batches in & out: ", raw_in_batch, ", ", raw_out_batch, "   ",norm)
	
	

	for test_iter := 0; test_iter < total_test_num; test_iter++ {
		fmt.Println(test_iter+1, "-th iter...start")
		raw_input := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_in_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*raw_in_batch)
		ker_in := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_ker_"+strconv.Itoa(test_iter)+".csv", raw_in_batch*raw_out_batch*ker_wid*ker_wid)
		bn_a := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_bna_"+strconv.Itoa(test_iter)+".csv", raw_out_batch)
		bn_b := readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_bnb_"+strconv.Itoa(test_iter)+".csv", raw_out_batch)

		// input encryption
		//input := prep_Input(raw_input, raw_in_wid, in_wid, cont.N, norm, trans, printResult)
		input1 := prep_Input_xuchao(raw_input, split_batch, raw_in_wid, in_wid, cont.N, norm, trans, printResult)
					
		//start := time.Now()		
		ctxt_input := make([]*ckks.Ciphertext, split_batch)				
						
		plain_tmp := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		for i := 0; i < split_batch; i++ {
			cont.encoder.EncodeCoeffs(input1[i], plain_tmp)
			ctxt_input[i] = cont.encryptor.EncryptNew(plain_tmp)
			
		}
				
		// Kernel Prep & Conv (+BN) Evaluation
		
			
		var ct_result []*ckks.Ciphertext
		//if boot {		
		ct_result = evalConv_BNRelu_new_xuchao(cont, ctxt_input, ker_in, bn_a, bn_b, 0.0, pow, split_batch, in_wid, kp_wid, ker_wid, raw_in_batch, raw_out_batch, norm, 0, 0, 2, 0, kind, false, false)
								
		//}
		
		cfs_tmp_add := make([][]float64, split_batch)
		test_out := make([]float64, split_batch * cont.N)
		start := time.Now()
		
		for i := 0; i < split_batch; i++ {
			cont.decryptor.Decrypt(ct_result[i], plain_tmp)
			cfs_tmp_add[i] = cont.encoder.DecodeCoeffs(plain_tmp)
			cfs_tmp_add[i] = post_process(cfs_tmp_add[i], raw_in_wid, in_wid)
			
		}
		
		
		delta := raw_out_batch/split_batch		
		for i := 0; i < split_batch; i++ {
			for j := 0; j < len(cfs_tmp_add[i]); j++ {
			
				idx := (int)(j/delta)				
				test_out[idx * raw_in_batch + i * delta + j % delta ] = cfs_tmp_add[i][j]
			}
			
		}
		
		
		
		fmt.Printf("Decryption Done in %s \n", time.Since(start))
		//test_out := post_process(cfs_tmp, raw_in_wid, in_wid)
		var real_out []float64
		if boot {
			real_out = readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_reluout_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*raw_in_batch)
		} else {
			real_out = readTxt(test_dir+"test_conv"+strconv.Itoa(ker_wid)+"_batch_"+strconv.Itoa(in_batch)+"_out_"+strconv.Itoa(test_iter)+".csv", raw_in_wid*raw_in_wid*raw_in_batch)
		}

		printDebugCfsPlain(test_out, real_out)
		
	}

}


func testResNet_xuchao(st, end, ker_wid, depth, split int, debug, cf100 bool) {
	
	split = 6
	split_batch := 1 << split

	// init_batch fixed to 16
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/" // !! NEED to remove "_test"
	//out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
	fc_out := 10    // 100 for cifar100
	init_pow := 6.0 // covers [-2^pow, 2^pow] values at ReLU evaluation
	mid_pow := 6.0
	final_pow := 6.0
	
	if cf100 {
		weight_dir = "Resnet_weights/weights_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		//out_dir = "Resnet_enc_results/results_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		//fc_out = 100 // 100 for cifar100
		if ker_wid == 3 {
			final_pow = 7.0
		} else if ker_wid == 5 {
			final_pow = 6.0
		} else {
			final_pow = 5.0
		}
		init_pow = 5.0
		mid_pow = 5.0
	}
	
	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth (not in 8, 14, 20)!")
	}
	real_batch := []int{16, 32, 64} // same as python (small for faster eval test) !! NEEDS to be changed for real test input {16, 32, 64}
	norm := []int{4, 8, 16}         // only use 1/norm batches among full batches (i.e., sparse packing)
	//norm := []int{1, 1, 1}         // only use 1/norm batches among full batches (i.e., sparse packing)
	step := []int{1, 1, 1}          // non-one only when it is for inside

	logN := 16 - split // !! NEEDS to be modified to 16
	alpha := 0.0
	in_wids := []int{32, 16, 8}                                         // before cropping
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2} // same as python
	//fast_pack := true
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}

	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, "Conv")

	for iter := st; iter < end; iter++ {
		fmt.Println("Running ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		// image := make([]float64, in_wids[0]*in_wids[0]*3)
		// for i := range image {
		// 	image[i] = 1.0 - 1.0*float64(i)/float64(len(image))
		// }
		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		
		input := make([][]float64, 3)
		for i := 0; i < 3; i++ {
			input[i] = make([]float64, cont.N)
		}
		
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				/*
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k] // sparse pack the input
					}
					k++
				}
				*/
				
				for b := 0; b < 3; b++ {				
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[b][i*in_wids[0]*max_batch[0]+j*max_batch[0]] = image[k]
						//input[b][i*in_wids[0]*max_batch[0]+j*max_batch[0] + norm[0]] = 1.0
																	
					}
					k++					
									
				}				
			}
		}
		
			
		fmt.Println("Input: ")
		//prt_mat_norm(input, max_batch[0], norm[0], 3, false)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)
		

		enc_start := time.Now()
		
		ct_input := make([]*ckks.Ciphertext, 3)
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		for b := 0; b < 3; b++ {
			cont.encoder.EncodeCoeffs(input[b], pl_input)
			ct_input[b] = cont.encryptor.EncryptNew(pl_input)
		}		
		
		fmt.Printf("Encryption done in %s \n", time.Since(enc_start))

		//timings := make([]float64, 6)
		//begin_start := time.Now()
		start := time.Now()

		// ResNet Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blcs[0]; i++ {
		//for i := 1; i <= 2; i++ {
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", real_batch[0])
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", real_batch[0])
			// bn_a := make([]float64, real_batch[0])
			// bn_b := make([]float64, real_batch[0])
			// for i := range bn_a {
			// 	bn_a[i] = 0.2
			// 	bn_b[i] = 0.0
			// }
			ker_in_batch := 3
			if i != 1 {
				ker_in_batch = real_batch[0]
			}
			ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", ker_in_batch*real_batch[0]*ker_size)
			// ker_in := make([]float64, ker_in_batch*real_batch[0]*ker_size)
			// for i := range ker_in {
			// 	ker_in[i] = 0.05 * float64(i) / float64(len(ker_in))
			// }
			ct_layer = evalConv_BNRelu_new_resnet_xuchao(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, split_batch, in_wids[0], raw_in_wids[0], ker_wid, ker_in_batch, real_batch[0], 1, 0, step[0], 2, 0, "Conv", false, debug)
						
			pow = mid_pow
			fmt.Println("Block1, Layer ", i, "done!",ct_layer[0].Level())
		}
		
		final_pow = final_pow
		
		res := make([][]float64, real_batch[0])
		for i := 0; i < real_batch[0]; i++ {
				cont.decryptor.Decrypt(ct_layer[i], pl_input)
				res[i] = cont.encoder.DecodeCoeffs(pl_input)
		}
		res_tmp := make([]float64, real_batch[0] * len(res[0]))
		kk := 0
		for j := 0; j < len(res[0]); j++ {
			for i := 0; i < real_batch[0]; i++ {
				res_tmp[kk] = res[i][j]
				kk++  				
			}			
		}		
				
		
		
		
		
		
		
		fmt.Println("Block1 done.") // !!!! HERE is DONE
		//timings[0] = time.Since(start).Seconds()
		//start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])
		// ker_in12 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
		// for i := range ker_in12 {
		// 	ker_in12[i] = 0.05 * float64(i) / float64(len(ker_in12))
		// }
		// bn_a := make([]float64, real_batch[1])
		// bn_b := make([]float64, real_batch[1])
		// for i := range bn_a {
		// 	bn_a[i] = 0.1
		// 	bn_b[i] = 0.0
		// }
		
		ct_layer = evalConv_BNRelu_new_resnet_xuchao(cont, ct_layer, ker_in12, bn_a, bn_b, alpha, pow, split_batch, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1], 1, 0, 4, 2, 0, "StrConv_sparse", false, debug)
		fmt.Println("Block1 to 2 done!")
		//timings[1] = time.Since(start).Seconds()
		//start = time.Now()
		
		
		
		
		res = make([][]float64, real_batch[1])
		for i := 0; i < real_batch[1]; i++ {
				cont.decryptor.Decrypt(ct_layer[i], pl_input)
				res[i] = cont.encoder.DecodeCoeffs(pl_input)
		}
		res_tmp = make([]float64, real_batch[1] * len(res[0]))
		
		kk = 0		
		for j := 0; j < len(res[0]); j++ {
			for i := 0; i < real_batch[1]; i++ {
				res_tmp[kk] = res[i][j]
				kk++  				
			}			
		}
						
		

		kk = 0		
		res_tmpxxxx := make([]float64, len(res[0]))
		for i := 0; i < len(res[0]); i++ {
			res_tmpxxxx[kk] = res[0][i]
			kk++  				
		}			
		
						
		
		
		
		
		//pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
			
		// ResNet Block 2
		//for i := 1; i <= num_blcs[1]; i++ {
		for i := 1; i <= num_blcs[1]; i++ {
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
			//bn_a2 := make([]float64, real_batch[1])
			//bn_b2 := make([]float64, real_batch[1])
			//ker_in2 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
			/*
			for i := range bn_a2 {
			 	bn_a2[i] = 1			 	
			}
			
			for i := range bn_b2 {			 	
			 	bn_b2[i] = 0.0
			}
			for i := range ker_in2 {				
			 	ker_in2[i] = 1.0
			 				 	
			}
			*/
			//ker_in2[0] = 1.0
				
			// for i := range ker_in2 {
			// 	ker_in2[i] = 0.05 * float64(i) / float64(len(ker_in2))
			// }
			//name = "Block2_" + strconv.Itoa(i)
			
						
			ct_layer = evalConv_BNRelu_new_resnet_xuchao(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, split_batch, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], 4, 0, step[0], 2, 2, "conv", false, debug)
			fmt.Println("Block2, Layer ", i, "done!", ct_layer[0].Level())
		}
		
		res = make([][]float64, real_batch[1])
		for i := 0; i < real_batch[1]; i++ {
				cont.decryptor.Decrypt(ct_layer[i], pl_input)
				res[i] = cont.encoder.DecodeCoeffs(pl_input)
		}
		res_tmp = make([]float64, real_batch[1] * len(res[0]))
		kk = 0
		
		for j := 0; j < len(res[0]); j++ {
			for i := 0; i < real_batch[1]; i++ {
				res_tmp[kk] = res[i][j]
				kk++  				
			}			
		}		
		/*
		
		for i := 0; i < len(res[0]); i++ {
			if i % 4 == 0 {
				res_tmp[kk] = res[0][i]
				kk++
			}  				
		}			
		*/	
			
		
				
		fmt.Println("Block2 done.")
		
		ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])
		// bn_a3 := make([]float64, real_batch[2])
		// bn_b3 := make([]float64, real_batch[2])
		// ker_in23 := make([]float64, real_batch[1]*real_batch[2]*ker_size)
		// for i := range bn_a3 {
		// 	bn_a3[i] = 0.1
		// 	bn_b3[i] = 0.0
		// }
		// for i := range ker_in23 {
		// 	ker_in23[i] = 0.05 * float64(i) / float64(len(ker_in23))
		// }
		//name = "Block2_to_3"
		ct_layer = evalConv_BNRelu_new_resnet_xuchao(cont, ct_layer, ker_in23, bn_a3, bn_b3, alpha, pow, split_batch, in_wids[1], raw_in_wids[2], ker_wid, real_batch[1], real_batch[2], 4, 0, 16, 2, 0, "StrConv_sparse", false, debug)
		fmt.Println("Block2 to 3 done!")
		//timings[3] = time.Since(start).Seconds()
		//start = time.Now()
		
		res = make([][]float64, real_batch[2])
		for i := 0; i < real_batch[2]; i++ {
				cont.decryptor.Decrypt(ct_layer[i], pl_input)
				res[i] = cont.encoder.DecodeCoeffs(pl_input)
		}
		res_tmp = make([]float64, real_batch[2] * len(res[0]))
		kk = 0
		
		for j := 0; j < len(res[0]); j++ {
			for i := 0; i < real_batch[2]; i++ {
				res_tmp[kk] = res[i][j]
				kk++  				
			}			
		}		
		
		/*
		for i := 0; i < len(res[0]); i++ {
			if i % 16 == 0 {		
				res_tmp[kk] = res[0][i]
				kk++
			}			 				
		}			
		*/	
			
		
		
		
		
		
		
		
		// ResNet Block 3
		for i := 1; i <= num_blcs[2]; i++ {
		//for i := 1; i <= 1; i++ {
			bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
			bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
			ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)
			// bn_a3 := make([]float64, real_batch[2])
			// bn_b3 := make([]float64, real_batch[2])
			// ker_in3 := make([]float64, real_batch[2]*real_batch[2]*ker_size)
			// for i := range bn_a3 {
			// 	bn_a3[i] = 0.1
			// 	bn_b3[i] = 0.0
			// }
			// for i := range ker_in3 {
			// 	ker_in3[i] = 0.1 * float64(i) / float64(len(ker_in3))
			// }

			if i == num_blcs[2] {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new_resnet_xuchao(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, pow, split_batch, in_wids[2], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], 16, 0, step[0], 2, 4, "Conv", false, debug)
			fmt.Println("Block3, Layer ", i, "done!", ct_layer[0].Level())
		}
		
		res = make([][]float64, real_batch[2])
		for i := 0; i < real_batch[2]; i++ {
				cont.decryptor.Decrypt(ct_layer[i], pl_input)
				res[i] = cont.encoder.DecodeCoeffs(pl_input)
		}
		res_tmp = make([]float64, real_batch[2] * len(res[0]))
		kk = 0
		
		for j := 0; j < len(res[0]); j++ {
			for i := 0; i < real_batch[2]; i++ {
				res_tmp[kk] = res[i][j]
				kk++  				
			}			
		}		
		
		/*
		for i := 0; i < len(res[0]); i++ {
					
			res_tmp[kk] = res[0][i]
			kk++
						 				
		}			
		*/	
			
		
		
		
		fmt.Println("Block3 done.")
		//timings[4] = time.Since(start).Seconds()
		//start = time.Now()

		ker_inf_wid := raw_in_wids[2]
		if ker_inf_wid%2 == 0 {
			ker_inf_wid++
		}
		ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)
		// ker_inf := make([]float64, real_batch[2]*fc_out)
		// for i := range ker_inf {
		// 	ker_inf[i] = 0.1 * float64(i)
		// }
		
		var ct_result []*ckks.Ciphertext
		if cf100 {
						
			fmt.Println("Final FC done.")
			//timings[5] = time.Since(start).Seconds()
			//start = time.Now()
		} else {
			ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
			for i := range ker_inf {
				for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
					ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
				}
			}
			bn_af := make([]float64, fc_out)
			for i := range bn_af {
				bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
			}
			bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			// bn_bf := make([]float64, fc_out)
			// for i := range bn_bf {
			// 	bn_bf[i] = 1 * float64(i)
			// }
			//ct_result = evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[2], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false)
			
			ct_result = evalConv_BN_resnet_xuchao(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[2], ker_inf_wid, real_batch[2], fc_out, 16, split_batch, float64(1<<30), false)

			
			fmt.Println("Final FC done.")
			//timings[5] = time.Since(start).Seconds()
			//start = time.Now()
		}
		
		
		
		
		
		fmt.Println()
		fmt.Println("===============  DECRYPTION  ===============")
		fmt.Println()
		if cf100 {
			
			fmt.Println("\n result: ")
			//writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out)
		} else {
			res := make([][]float64, fc_out)
			for i := 0; i < fc_out; i++ {
				cont.decryptor.Decrypt(ct_result[i], pl_input)
				res[i] = cont.encoder.DecodeCoeffs(pl_input)
				//res_out := prt_mat_one_norm(res_tmp, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
				//fmt.Println("\n result: ", res_out[:fc_out])
				//writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])
			}
			res_tmp := make([]float64, fc_out * len(res[0]))
			k := 0
			for i := 0; i < len(res[0]); i++ {
				for j := 0; j < fc_out; j++ {
					res_tmp[k] = res[j][i]
					k++  				
				}			
			}
			
			res_tmp_tr := make([]float64, 65536)
			
			for i := 0; i < len(res_tmp); i++ {
				if i % 160 < 10 {
					idx := (i % 160) * 16 + (int)(i / 160) * 1024  
					res_tmp_tr[idx] = res_tmp[i]
				}
			
			}
			
			
			
			res_out := prt_mat_one_norm(res_tmp_tr, 1024, norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
			fmt.Println("\n result: ", res_out[:fc_out])
			fmt.Printf("Decryption Done in %s \n", time.Since(start))
		}
		/*
		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Blc2->3: ", timings[3], " sec")
		fmt.Println("Blc3: ", timings[4], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
		
		fmt.Printf("xuchao split Total done in %s \n", time.Since(begin_start))
		*/
	}

}

func megre_multipic(cont *context, start, merge_num int, ct_input []*ckks.Ciphertext)(ct_res *ckks.Ciphertext){
	
	
	xi_plain := ckks.NewPlaintext(cont.params, cont.ECD_LV, 1.0)
	
	ct_res = ct_input[start]
	
	for i := 1; i < merge_num; i++ {
	
		inputx := make([]float64, cont.N)
		inputx[i] = 1.0   
		cont.encoder.EncodeCoeffs(inputx, xi_plain)
		cont.encoder.ToNTT(xi_plain)
		
		tmp_ct_layer1 := cont.evaluator.MulNew(xi_plain,ct_input[start + i])		
		ct_res = cont.evaluator.AddNew(tmp_ct_layer1,ct_res)
			
	}
	return ct_res
}


func move_multipic(cont *context, start, num int, ct_input []*ckks.Ciphertext)(ct_res []*ckks.Ciphertext){
	
	
	xi_plain := ckks.NewPlaintext(cont.params, cont.ECD_LV, 1.0)
	
	//ct_res = make([]*ckks.Ciphertext, num)
	//ct_res[start] = ct_input[start]
	
	for i := 1; i < num; i++ {
	
		inputx := make([]float64, cont.N)
		inputx[cont.N - i] = -1.0   
		cont.encoder.EncodeCoeffs(inputx, xi_plain)
		cont.encoder.ToNTT(xi_plain)
		
		ct_input[i + start] = cont.evaluator.MulNew(xi_plain,ct_input[i + start])		
		
			
	}
	
	
	
	return ct_input
}




func split_multipic(cont *context, ct_input *ckks.Ciphertext, num int)(ct_res []*ckks.Ciphertext){
	
	ct_res = make([]*ckks.Ciphertext, num)
	xi_plain := ckks.NewPlaintext(cont.params, cont.ECD_LV, 1.0)
	
	
	
	inputx := make([]float64, cont.N) 
	cont.encoder.EncodeCoeffs(inputx, xi_plain)
	cont.encoder.ToNTT(xi_plain)
	for i := 1; i < num; i++ {
		ct_res[i] = cont.evaluator.MulNew(xi_plain,ct_input)
	}
	
	inputx[0] = 1
	cont.encoder.EncodeCoeffs(inputx, xi_plain)
	cont.encoder.ToNTT(xi_plain)
	ct_res[0] = cont.evaluator.MulNew(xi_plain,ct_input)
		
	for i := 1; i < cont.N; i++ {
		// [1,2,3,4]->[2,3,4,1]
		inputx = make([]float64, cont.N)
		inputx[cont.N - i] = -1   
		cont.encoder.EncodeCoeffs(inputx, xi_plain)
		cont.encoder.ToNTT(xi_plain)		
		tmp := cont.evaluator.MulNew(xi_plain,ct_input)
		
		// [2,3,4,1]->[2,0,0,0]
		inputx[cont.N - i] = 0  
		inputx[0] = 1  
		cont.encoder.EncodeCoeffs(inputx, xi_plain)
		cont.encoder.ToNTT(xi_plain)		
		tmp1 := cont.evaluator.MulNew(xi_plain,tmp)
		
		// [2,0,0,0]->[0,2,0,0]
		idx := int(i/num)
		inputx[0] = 0
		inputx[idx] = 1
		cont.encoder.EncodeCoeffs(inputx, xi_plain)
		cont.encoder.ToNTT(xi_plain)	
		tmp2 := cont.evaluator.MulNew(xi_plain,tmp1)	
		ct_res[i % num] = cont.evaluator.AddNew(ct_res[i % num],tmp2)
		
	}	
				
	return ct_res
}


func testResNet_crop_sparse_multipic_BL(st, end, ker_wid, depth int, debug, cf100 bool) {
	// init_batch fixed to 16
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/" // !! NEED to remove "_test"
	//out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
	//fc_out := 10    // 100 for cifar100
	//init_pow := 6.0 // covers [-2^pow, 2^pow] values at ReLU evaluation
	//mid_pow := 6.0
	//final_pow := 6.0
	/*
	if cf100 {
		weight_dir = "Resnet_weights/weights_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		out_dir = "Resnet_enc_results/results_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		fc_out = 100 // 100 for cifar100
		if ker_wid == 3 {
			final_pow = 7.0
		} else if ker_wid == 5 {
			final_pow = 6.0
		} else {
			final_pow = 5.0
		}
		init_pow = 5.0
		mid_pow = 5.0
	}
	*/
	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth (not in 8, 14, 20)!")
	}
	real_batch := []int{16, 32, 64} // same as python (small for faster eval test) !! NEEDS to be changed for real test input {16, 32, 64}
	//norm := []int{1, 1, 1}         // only use 1/norm batches among full batches (i.e., sparse packing)
	//step := []int{1, 1, 1}          // non-one only when it is for inside
	pad := ker_wid / 2
	logN := 16 // !! NEEDS to be modified to 16
	//alpha := 0.0
	in_wids := []int{32, 16, 8} 
	//in_wids := []int{32,32,32}                                         // before cropping
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2} // same as python
	//fast_pack := true
	ker_size := ker_wid * ker_wid
	/*
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}
	*/
	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, "BL_Conv")
	//return 
	name := ""
	ct_input := make([]*ckks.Ciphertext, end)
	//pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	
	N := 16384
	
	enc_start_xuchao := time.Now()
	ct_layer := ct_input
	//pow := init_pow
	
	for iter := st; iter < end; iter++ {
		fmt.Println("Running ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(0)+".csv", in_wids[0]*in_wids[0]*3)
		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		input := make([]float64, len(image))
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						//input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k] // sparse pack the input
						//input[i*in_wids[0]*3+j*3+b*norm[0]] = image[k] // sparse pack the input
						input[k] =  image[k]
						
					}
					k++
				}
			}
		}
		
		
		
		
		
		fmt.Println("Input: ")
		//prt_mat_norm(input, max_batch[0], norm[0], 3, false)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)
		enc_start := time.Now()
		//pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		input_rs := reshape_input_BL_xc(input, in_wids[0])
		
		input1_rs_tmp := make([]complex128, 2 * N)
		
		input1xc := make([]float64, len(input_rs) * 2)
				
		for i := 0; i < len(input_rs); i++ {
			input1xc[i] = real(input_rs[i])
			input1_rs_tmp[i] = input_rs[i]
			input1_rs_tmp[i + N] = input_rs[i]
				
		}
		
		
		ct_layer[iter] = cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input1_rs_tmp, cont.logN-1))
		
		
		
		fmt.Printf("Encryption done in %s \n", time.Since(enc_start))

		//timings := make([]float64, 6)
		//begin_start := time.Now()
		//sstart := time.Now()
	}
	pow := 4.0
	alpha := 0.0
	//for i := 1; i <= 1; i++ {
	
	for i := 1; i <= num_blcs[0]; i++ {
	
		// ResNet Block 1
				
		for iter := st; iter < end; iter++ {
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", real_batch[0])
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", real_batch[0])
			
			//bn_a := make([]float64, real_batch[0])
			//bn_b := make([]float64, real_batch[0])
			//for i := range bn_a {
			//		bn_a[i] = 1.0
			//	bn_b[i] = 0.0
			//}
			ker_in_batch := 3
			if i != 1 {
				ker_in_batch = real_batch[0]
			}
			ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", ker_in_batch*real_batch[0]*ker_size)
			
			//ker_in2 := make([]float64, 2*N)
			//for i := range ker_in {
			// 	ker_in2[i] = ker_in[i]
			// 	ker_in2[N+i] = ker_in[i]
			//}
			
			if i != 1 {
				img := readTxt("TestWant_xuchao/Block1_"+strconv.Itoa(i)+"_valuesTest", in_wids[0]*in_wids[0]*real_batch[0])
				input := make([]float64, len(img))
				k := 0
				for i := 0; i < in_wids[0]; i++ {
					for j := 0; j < in_wids[0]; j++ {
						for b := 0; b < real_batch[0]; b++ {
							if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
								input[k] =  img[k]
						
							}
							k++
						}
					}
				}
				input_rs := reshape_input_BL_xc(input, in_wids[0])		
				input1_rs_tmp := make([]complex128, 2 * N)		
				input1xc := make([]float64, len(input_rs) * 2)
				
				for i := 0; i < len(input_rs); i++ {
					input1xc[i] = real(input_rs[i])
					input1_rs_tmp[i] = input_rs[i]
					input1_rs_tmp[i + N] = input_rs[i]
				
				}
				ct_layer[iter] = cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input1_rs_tmp, cont.logN-1))
			
			}
			
			
			name = "Block1_" + strconv.Itoa(i)
			
			
			
			
			start := time.Now()
			ct_layer[iter] = evalConv_BN_BL_test_xc(cont, ct_layer[iter], ker_in, bn_a, bn_b, in_wids[0], ker_wid, ker_in_batch, real_batch[0], 0, 1, pad, 2, false, false)
			//ct_layer[iter] = evalConv_BN_BL_test(cont, ct_layer[iter], ker_in, bn_a, bn_b, in_wids[0], ker_wid, ker_in_batch, real_batch[0], 0, 1, pad,false, false)
			
			write_Ciphertext(cont,ct_layer[iter],logN-1,"BL_res/real_1_"+strconv.Itoa(i) +"_"+strconv.Itoa(iter))
			write_Ciphertext_image(cont,ct_layer[iter],logN-1,"BL_res/imag_1_"+strconv.Itoa(i) +"_"+strconv.Itoa(iter))
					
			ct_layer[iter].Scale = ct_layer[iter].Scale * math.Pow(2, pow+2)
			
			fmt.Println("ct_layer[iter] level = ", ct_layer[iter].Level())
						
			ct_boot := cont.btp.Bootstrapp(ct_layer[iter])
			
			pl_scale := ckks.NewPlaintext(cont.params, ct_boot.Level(), math.Pow(2, 30)*float64(cont.params.Q()[14])*float64(cont.params.Q()[13])/ct_boot.Scale)
			val_scale := make([]complex128, cont.N/2)
			for i := range val_scale {
				val_scale[i] = complex(1.0, 0) // val_scale[i] = complex(1.0/math.Pow(2, pow), 0)
			}
			cont.encoder.EncodeNTT(pl_scale, val_scale, cont.logN-1)
			cont.evaluator.Mul(ct_boot, pl_scale, ct_boot)
			cont.evaluator.Rescale(ct_boot, cont.params.Scale(), ct_boot)
			
			ct_res := make([]*ckks.Ciphertext, 2)
			ct_iboot := cont.pack_evaluator.ConjugateNew(ct_boot)
			ct_res[0] = cont.evaluator.AddNew(ct_boot, ct_iboot)
			ct_res[1] = cont.evaluator.DivByiNew(cont.evaluator.SubNew(ct_boot, ct_iboot))
			

			
			for pos := 0; pos < 2; pos++ {
				// fmt.Println("before relu: LV = ", ct_res[pos].Level(), " Scale = ", math.Log2(ct_res[pos].Scale))
				// fmt.Println("after Rescale: LV = ", ct_boot.Level(), " Scale = 2^", math.Log2(ct_boot.Scale))
				ct_res[pos] = evalReLU(cont.params, cont.evaluator, ct_res[pos], alpha)
				cont.evaluator.MulByPow2(ct_res[pos], int(pow), ct_res[pos])
				cont.evaluator.SetScale(ct_res[pos], cont.params.Scale())
				// printDebug(cont.params, ct_res[pos], vals_relu, cont.decryptor, cont.encoder)
			}
			
			ct_layer[iter] = cont.evaluator.AddNew(ct_res[0],  cont.evaluator.MultByiNew(ct_res[1]))
			fmt.Printf("xuchao one BL boot and relu done in %s \n", time.Since(start))
			
						
							
		}
					
		//pow = mid_pow
		fmt.Println("Block1, Layer iter", i, "done!")
		
		fmt.Println("Block1 done.") // !!!! HERE is DONE
		
	}
	fmt.Printf("xuchao num_blcs[0] done in %s \n", time.Since(enc_start_xuchao))
	
	
	ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
	bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
	bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])
	name = "Block1_to_2"
	
	for i := st; i < 2; i++{
		for iter := st; iter < end; iter++ {
			
			
			img := readTxt("TestWant_xuchao/Block1_7_valuesTest", in_wids[0]*in_wids[0]*real_batch[0])
				input := make([]float64, len(img))
				k := 0
				for i := 0; i < in_wids[0]; i++ {
					for j := 0; j < in_wids[0]; j++ {
						for b := 0; b < real_batch[0]; b++ {
							if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
								input[k] =  img[k]
								
							}
							k++
						}
					}
				}
				input_rs := reshape_input_BL_xc(input, in_wids[0])		
				input1_rs_tmp := make([]complex128, 2 * N)		
				input1xc := make([]float64, len(input_rs) * 2)
				
				for i := 0; i < len(input_rs); i++ {
					input1xc[i] = real(input_rs[i])
					input1_rs_tmp[i] = input_rs[i]
					input1_rs_tmp[i + N] = input_rs[i]
				
				}
				ct_layer[iter] = cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input1_rs_tmp, cont.logN-1))
			
			
			
			
			
			ct_layer[iter] = evalConv_BN_BL_test(cont, ct_layer[iter], ker_in12, bn_a, bn_b, in_wids[0], ker_wid, real_batch[0], real_batch[1], 0, 1, pad, false, false)
			//write_Ciphertext(cont,ct_layer[iter],logN-1,"multipic_BL")
			ct_out := ct_layer[iter]
			for aa := 0; aa < in_wids[0] - 1; aa++{
				for bb := 0; bb < in_wids[0] - 1; bb++{
					if aa % 2 == 0 && bb % 2 == 0 {
						c_tmp := make([]complex128, N * 2)
						for cc := 0; cc < real_batch[1]; cc++{ 
							c_tmp[aa*in_wids[0]*real_batch[1] + bb*real_batch[1] + cc] = complex(1.0, 0)							
						}
						
						pl_tmp := ckks.NewPlaintext(cont.params, ct_layer[iter].Level(), cont.params.Scale())
						cont.encoder.Encode(pl_tmp, c_tmp, cont.params.LogSlots())
						cont.encoder.ToNTT(pl_tmp)
						
						if aa == 0 && bb == 0 {
							ct_out = cont.evaluator.MulNew(ct_layer[iter], pl_tmp)
						} else {
						
							//ct_out_t := cont.evaluator.MulNew(ct_layer[iter], pl_tmp)
							mv := aa*in_wids[0]*real_batch[1] + bb*real_batch[1] - (aa*in_wids[0]*real_batch[1]/2 + bb*real_batch[1]/2)
							
							ct_out_t := cont.pack_evaluator.RotateNew(cont.evaluator.MulNew(ct_layer[iter], pl_tmp), mv)
							
							cont.evaluator.Add(ct_out_t,ct_out,ct_out)							
						}
					}	
				}			
			}
			ct_layer[iter] = ct_out
			
			
			write_Ciphertext(cont,ct_layer[iter],logN-1,"BL_res/Block1_to_2real_1_"+strconv.Itoa(i) +"_"+strconv.Itoa(iter))
			write_Ciphertext_image(cont,ct_layer[iter],logN-1,"BL_res/Block1_to_2imag_1_"+strconv.Itoa(i) +"_"+strconv.Itoa(iter))
			
			
			ct_layer[iter].Scale = ct_layer[iter].Scale * math.Pow(2, pow+2)
			ct_boot := cont.btp.Bootstrapp(ct_layer[iter])
			
			pl_scale := ckks.NewPlaintext(cont.params, ct_boot.Level(), math.Pow(2, 30)*float64(cont.params.Q()[14])*float64(cont.params.Q()[13])/ct_boot.Scale)
			val_scale := make([]complex128, cont.N/2)
			for i := range val_scale {
				val_scale[i] = complex(1.0, 0) // val_scale[i] = complex(1.0/math.Pow(2, pow), 0)
			}
			cont.encoder.EncodeNTT(pl_scale, val_scale, cont.logN-1)
			cont.evaluator.Mul(ct_boot, pl_scale, ct_boot)
			cont.evaluator.Rescale(ct_boot, cont.params.Scale(), ct_boot)
			
			ct_res := make([]*ckks.Ciphertext, 2)
			ct_iboot := cont.pack_evaluator.ConjugateNew(ct_boot)
			ct_res[0] = cont.evaluator.AddNew(ct_boot, ct_iboot)
			ct_res[1] = cont.evaluator.DivByiNew(cont.evaluator.SubNew(ct_boot, ct_iboot))
			

			
			for pos := 0; pos < 2; pos++ {
				// fmt.Println("before relu: LV = ", ct_res[pos].Level(), " Scale = ", math.Log2(ct_res[pos].Scale))
				// fmt.Println("after Rescale: LV = ", ct_boot.Level(), " Scale = 2^", math.Log2(ct_boot.Scale))
				ct_res[pos] = evalReLU(cont.params, cont.evaluator, ct_res[pos], alpha)
				cont.evaluator.MulByPow2(ct_res[pos], int(pow), ct_res[pos])
				cont.evaluator.SetScale(ct_res[pos], cont.params.Scale())
				// printDebug(cont.params, ct_res[pos], vals_relu, cont.decryptor, cont.encoder)
			}
			
			ct_layer[iter] = cont.evaluator.AddNew(ct_res[0],  cont.evaluator.MultByiNew(ct_res[1]))
			//fmt.Printf("xuchao one BL boot and relu done in %s \n", time.Since(start))
		}			
	}
	
	fmt.Printf("xuchao Block1_to_2 done in %s \n", time.Since(enc_start_xuchao))
	
	
	for i := 1; i <= num_blcs[1]; i++ {
		bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
		bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
		ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
		
		for iter := st; iter < end/2; iter++ {
			img := readTxt("TestWant_xuchao/Block2_"+strconv.Itoa(i)+"_valuesTest", in_wids[1]*in_wids[1]*real_batch[1])
			input := make([]float64, len(img))
			k := 0
			for i := 0; i < in_wids[1]; i++ {
				for j := 0; j < in_wids[1]; j++ {
					for b := 0; b < real_batch[1]; b++ {
						if (i < raw_in_wids[1]) && (j < raw_in_wids[1]) {
							input[k] =  img[k]
						
						}
						k++
					}
				}
			}
			input_rs := reshape_input_BL_xc(input, in_wids[1])		
			input1_rs_tmp := make([]complex128, 2 * N)		
			input1xc := make([]float64, len(input_rs) * 2)
			
			
				
			for i := 0; i < len(input_rs); i++ {
				input1xc[i] = real(input_rs[i])
				input1_rs_tmp[i] = input_rs[i]
				input1_rs_tmp[i + N/2] = input_rs[i]
				input1_rs_tmp[i + 2*N/2] = input_rs[i]
				input1_rs_tmp[i + 3*N/2] = input_rs[i]
				
			}
			ct_layer[iter] = cont.encryptor.EncryptNew(cont.encoder.EncodeAtLvlNew(cont.ECD_LV, input1_rs_tmp, cont.logN-1))
		
		
			
		
			name = "Block2_" + strconv.Itoa(i)
			
			
			start := time.Now()
			ct_layer[iter] = evalConv_BN_BL_test_xc(cont, ct_layer[iter], ker_in2, bn_a2, bn_b2, in_wids[1], ker_wid, real_batch[1], real_batch[1], 0, 1, pad, 4, false, false)
			
			
			write_Ciphertext(cont,ct_layer[iter],logN-1,"BL_res/real_2_"+strconv.Itoa(i) +"_"+strconv.Itoa(iter))
			write_Ciphertext_image(cont,ct_layer[iter],logN-1,"BL_res/imag_2_"+strconv.Itoa(i) +"_"+strconv.Itoa(iter))
			
			//write_Ciphertext(cont,ct_layer[iter],logN-1,"multipic_BL")
			
			
			ct_layer[iter].Scale = ct_layer[iter].Scale * math.Pow(2, pow+2)
			ct_boot := cont.btp.Bootstrapp(ct_layer[iter])
			
			pl_scale := ckks.NewPlaintext(cont.params, ct_boot.Level(), math.Pow(2, 30)*float64(cont.params.Q()[14])*float64(cont.params.Q()[13])/ct_boot.Scale)
			val_scale := make([]complex128, cont.N/2)
			for i := range val_scale {
				val_scale[i] = complex(1.0, 0) // val_scale[i] = complex(1.0/math.Pow(2, pow), 0)
			}
			cont.encoder.EncodeNTT(pl_scale, val_scale, cont.logN-1)
			cont.evaluator.Mul(ct_boot, pl_scale, ct_boot)
			cont.evaluator.Rescale(ct_boot, cont.params.Scale(), ct_boot)
			
			ct_res := make([]*ckks.Ciphertext, 2)
			ct_iboot := cont.pack_evaluator.ConjugateNew(ct_boot)
			ct_res[0] = cont.evaluator.AddNew(ct_boot, ct_iboot)
			ct_res[1] = cont.evaluator.DivByiNew(cont.evaluator.SubNew(ct_boot, ct_iboot))
			

			
			for pos := 0; pos < 2; pos++ {
				// fmt.Println("before relu: LV = ", ct_res[pos].Level(), " Scale = ", math.Log2(ct_res[pos].Scale))
				// fmt.Println("after Rescale: LV = ", ct_boot.Level(), " Scale = 2^", math.Log2(ct_boot.Scale))
				ct_res[pos] = evalReLU(cont.params, cont.evaluator, ct_res[pos], alpha)
				cont.evaluator.MulByPow2(ct_res[pos], int(pow), ct_res[pos])
				cont.evaluator.SetScale(ct_res[pos], cont.params.Scale())
				// printDebug(cont.params, ct_res[pos], vals_relu, cont.decryptor, cont.encoder)
			}
			
			ct_layer[iter] = cont.evaluator.AddNew(ct_res[0],  cont.evaluator.MultByiNew(ct_res[1]))
			fmt.Printf("xuchao one BL boot and relu done in %s \n", time.Since(start))
						
				
		}
	}
	
	fmt.Printf("xuchao num_blcs[1] done in %s \n", time.Since(enc_start_xuchao))
	ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
	bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
	bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])	
	name = "Block2_to_3"
	
		for iter := st; iter < end/2; iter++ {
			
			
			
			
			ct_layer[iter] = evalConv_BN_BL_test(cont, ct_layer[iter], ker_in23, bn_a3, bn_b3, in_wids[1], ker_wid, real_batch[1], real_batch[2], 0, 1, pad, false, false)
			//write_Ciphertext(cont,ct_layer[iter],logN-1,"multipic_BL")
			
					
			
			ct_layer[iter].Scale = ct_layer[iter].Scale * math.Pow(2, pow+2)
			ct_boot := cont.btp.Bootstrapp(ct_layer[iter])
			
			pl_scale := ckks.NewPlaintext(cont.params, ct_boot.Level(), math.Pow(2, 30)*float64(cont.params.Q()[14])*float64(cont.params.Q()[13])/ct_boot.Scale)
			val_scale := make([]complex128, cont.N/2)
			for i := range val_scale {
				val_scale[i] = complex(1.0, 0) // val_scale[i] = complex(1.0/math.Pow(2, pow), 0)
			}
			cont.encoder.EncodeNTT(pl_scale, val_scale, cont.logN-1)
			cont.evaluator.Mul(ct_boot, pl_scale, ct_boot)
			cont.evaluator.Rescale(ct_boot, cont.params.Scale(), ct_boot)
			
			ct_res := make([]*ckks.Ciphertext, 2)
			ct_iboot := cont.pack_evaluator.ConjugateNew(ct_boot)
			ct_res[0] = cont.evaluator.AddNew(ct_boot, ct_iboot)
			ct_res[1] = cont.evaluator.DivByiNew(cont.evaluator.SubNew(ct_boot, ct_iboot))
			

			
			for pos := 0; pos < 2; pos++ {
				// fmt.Println("before relu: LV = ", ct_res[pos].Level(), " Scale = ", math.Log2(ct_res[pos].Scale))
				// fmt.Println("after Rescale: LV = ", ct_boot.Level(), " Scale = 2^", math.Log2(ct_boot.Scale))
				ct_res[pos] = evalReLU(cont.params, cont.evaluator, ct_res[pos], alpha)
				cont.evaluator.MulByPow2(ct_res[pos], int(pow), ct_res[pos])
				cont.evaluator.SetScale(ct_res[pos], cont.params.Scale())
				// printDebug(cont.params, ct_res[pos], vals_relu, cont.decryptor, cont.encoder)
			}
			
			ct_layer[iter] = cont.evaluator.AddNew(ct_res[0],  cont.evaluator.MultByiNew(ct_res[1]))
			//fmt.Printf("xuchao one BL boot and relu done in %s \n", time.Since(start))
		}			
		
	fmt.Printf("xuchao num_blcs[1] done in %s \n", time.Since(enc_start_xuchao))
	
	for i := 1; i <= num_blcs[2]; i++ {
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
		ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)
		
		for iter := st; iter < end/4; iter++ {
			name = "Block2_" + strconv.Itoa(i)
			
			
			start := time.Now()
			ct_layer[iter] = evalConv_BN_BL_test(cont, ct_layer[iter], ker_in3, bn_a3, bn_b3, in_wids[2], ker_wid, real_batch[2], real_batch[2], 0, 1, pad, false, false)
			//write_Ciphertext(cont,ct_layer[iter],logN-1,"multipic_BL")
			
			
			ct_layer[iter].Scale = ct_layer[iter].Scale * math.Pow(2, pow+2)
			ct_boot := cont.btp.Bootstrapp(ct_layer[iter])
			
			pl_scale := ckks.NewPlaintext(cont.params, ct_boot.Level(), math.Pow(2, 30)*float64(cont.params.Q()[14])*float64(cont.params.Q()[13])/ct_boot.Scale)
			val_scale := make([]complex128, cont.N/2)
			for i := range val_scale {
				val_scale[i] = complex(1.0, 0) // val_scale[i] = complex(1.0/math.Pow(2, pow), 0)
			}
			cont.encoder.EncodeNTT(pl_scale, val_scale, cont.logN-1)
			cont.evaluator.Mul(ct_boot, pl_scale, ct_boot)
			cont.evaluator.Rescale(ct_boot, cont.params.Scale(), ct_boot)
			
			ct_res := make([]*ckks.Ciphertext, 2)
			ct_iboot := cont.pack_evaluator.ConjugateNew(ct_boot)
			ct_res[0] = cont.evaluator.AddNew(ct_boot, ct_iboot)
			ct_res[1] = cont.evaluator.DivByiNew(cont.evaluator.SubNew(ct_boot, ct_iboot))
			

			
			for pos := 0; pos < 2; pos++ {
				// fmt.Println("before relu: LV = ", ct_res[pos].Level(), " Scale = ", math.Log2(ct_res[pos].Scale))
				// fmt.Println("after Rescale: LV = ", ct_boot.Level(), " Scale = 2^", math.Log2(ct_boot.Scale))
				ct_res[pos] = evalReLU(cont.params, cont.evaluator, ct_res[pos], alpha)
				cont.evaluator.MulByPow2(ct_res[pos], int(pow), ct_res[pos])
				cont.evaluator.SetScale(ct_res[pos], cont.params.Scale())
				// printDebug(cont.params, ct_res[pos], vals_relu, cont.decryptor, cont.encoder)
			}
			
			ct_layer[iter] = cont.evaluator.AddNew(ct_res[0],  cont.evaluator.MultByiNew(ct_res[1]))
			fmt.Printf("xuchao one BL boot and relu done in %s \n", time.Since(start))			
				
		}
	}
	fmt.Printf("xuchao num_blcs[2] done in %s \n", time.Since(enc_start_xuchao))
	
	
	/*
	res := make([]float64, 65536)
				
	cont.decryptor.Decrypt(ct_layer[0], pl_input)
	res = cont.encoder.DecodeCoeffs(pl_input)
		
	writeTxt("xuchao_multipic_Block1_to_2", res)
	
	for i := 1; i <= num_blcs[1]; i++ {
		bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
		bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
		ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
		
		for iter := st; iter < end; iter++ {
			// ker_in := make([]float64, ker_in_batch*real_batch[0]*ker_size)
			// for i := range ker_in {
			// 	ker_in[i] = 0.05 * float64(i) / float64(len(ker_in))
			// }
			
			name = "Block2_" + strconv.Itoa(i)
			fmt.Println("aaaaaaaaaaaaaaaaaaaaaaaa "+ name, iter)
			ct_layer[iter] = evalConv_BN_multipic(cont, ct_layer[iter], ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, 3, "Conv_sparse", fast_pack, debug, name)
									
		}
		
		gap := 8
		fo := end / gap
		if end % gap > 0 {
			fo = fo + 1
		}
		res_ct := make([]*ckks.Ciphertext, fo)
		for it := 0; it < fo; it++ {
			aaa := 0
			
			if end - gap * it >= gap {
				aaa = gap
				
			} else {
				aaa = end - gap * it
				
			}
			res_ct[it] = megre_multipic(cont, it * gap, aaa, ct_layer)
				//cont.evaluator.Rotate(ct_layer[1], 1, ct_layer[1])
				//conv := cont.evaluator.AddNew(ct_layer[0], ct_layer[1])
			res_ct_tmp := make([]*ckks.Ciphertext, aaa)	
			res_ct_tmp = evalConv_Rule_multipic(cont, res_ct[it], alpha, pow, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], norm[1], 0, step[1], 2, 0, aaa,"Conv_sparse", fast_pack, debug, name)
			
			for ita := 0; ita < aaa; ita++ {
				ct_layer[it * gap + ita] = res_ct_tmp[ita]		
			}
			
			ct_layer = move_multipic(cont, gap * it, aaa, ct_layer)
								
		}
		
		res := make([]float64, 65536)				
		cont.decryptor.Decrypt(ct_layer[0], pl_input)
		res = cont.encoder.DecodeCoeffs(pl_input)
		writeTxt("xuchao_multipic_2_0_" + strconv.Itoa(i),res)
						
	}
	
	
	fmt.Printf("xuchao num_blcs[1] done in %s \n", time.Since(enc_start_xuchao))
	
	ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
	bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
	bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])	
	name = "Block2_to_3"
	
	
	for iter := st; iter < end; iter++ {			
		fmt.Println("aaaaaaaaaaaaaaaaaaaaaaaa "+ name, iter)
		ct_layer[iter] = evalConv_BNRelu_new_write(cont, ct_layer[iter], ker_in23, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[1], real_batch[2], norm[2], 0, step[2], 2, 2, "StrConv_sparse", fast_pack, debug, name)
		//ct_layer[iter] = evalConv_BN_multipic(cont, ct_layer[iter], ker_in23, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[1], real_batch[2], norm[2], 0, step[2], 2, 2, "StrConv_sparse", fast_pack, debug, name)
		
			
	}
		
	res = make([]float64, 65536)				
	cont.decryptor.Decrypt(ct_layer[0], pl_input)
	res = cont.encoder.DecodeCoeffs(pl_input)	
	writeTxt("xuchao_multipic_Block2_to_3", res)

		
	for i := 1; i <= num_blcs[2]; i++ {
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
		ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)
		if i == num_blcs[2] {
				pow = final_pow
		}
		for iter := st; iter < end; iter++ {
			// ker_in := make([]float64, ker_in_batch*real_batch[0]*ker_size)
			// for i := range ker_in {
			// 	ker_in[i] = 0.05 * float64(i) / float64(len(ker_in))
			// }
			
			name = "Block3_" + strconv.Itoa(i)
			fmt.Println("aaaaaaaaaaaaaaaaaaaaaaaa "+ name, iter)
			ct_layer[iter] = evalConv_BN_multipic(cont, ct_layer[iter], ker_in3, bn_a3, bn_b3, alpha, pow, in_wids[2], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 4, "Conv_sparse", fast_pack, debug, name)
									
		}
		
		gap := 16			
		
		fo := end / gap
		if end % gap > 0 {
			fo = fo + 1
		}
		res_ct := make([]*ckks.Ciphertext, fo)
		for it := 0; it < fo; it++ {
			aaa := 0
			
			if end - gap * it >= gap {
				aaa = gap
				
			} else {
				aaa = end - gap * it
				
			}
			res_ct[it] = megre_multipic(cont, it * 16, aaa, ct_layer)
				//cont.evaluator.Rotate(ct_layer[1], 1, ct_layer[1])
				//conv := cont.evaluator.AddNew(ct_layer[0], ct_layer[1])
			res_ct_tmp := make([]*ckks.Ciphertext, aaa)	
			res_ct_tmp = evalConv_Rule_multipic(cont, res_ct[it], alpha, pow, in_wids[2], raw_in_wids[2], ker_wid, real_batch[2], norm[2], 0, step[2], 2, 0, aaa,"Conv_sparse", fast_pack, debug, name)
			
			
			for ita := 0; ita < aaa; ita++ {
				ct_layer[it * gap + ita] = res_ct_tmp[ita]		
			}
			
			ct_layer = move_multipic(cont, gap * it, aaa, ct_layer)
								
		}
		
		res := make([]float64, 65536)				
		cont.decryptor.Decrypt(ct_layer[0], pl_input)
		res = cont.encoder.DecodeCoeffs(pl_input)
		writeTxt("xuchao_multipic_3_0_" + strconv.Itoa(i),res)
						
	}
	fmt.Printf("xuchao num_blcs[2] done in %s \n", time.Since(enc_start_xuchao))

	ker_inf_wid := raw_in_wids[2]
	if ker_inf_wid%2 == 0 {
		ker_inf_wid++
	}
	ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)
	ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
	for i := range ker_inf {
		for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
			ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
		}
	}
	bn_af := make([]float64, fc_out)
	for i := range bn_af {
		bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
	}
	bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			// bn_bf := make([]float64, fc_out)
			// for i := range bn_bf {
			// 	bn_bf[i] = 1 * float64(i)
			// }
	for iter := st; iter < end; iter++ {
		ct_layer[iter] = evalConv_BN(cont, ct_layer[iter], ker_inf_, bn_af, bn_bf, in_wids[2], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false)
	}
	
	
	
	fmt.Printf("xuchao Encryption done in %s \n", time.Since(enc_start_xuchao))

	for iter := st; iter < end; iter++ {
		cont.decryptor.Decrypt(ct_layer[iter], pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
			
		res_out := prt_mat_one_norm(res_tmp, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
		fmt.Println("\n result: ", res_out[:fc_out])
		writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])
	}

	*/
	

}



func testResNet_crop_sparse_multipic(st, end, ker_wid, depth int, debug, cf100 bool) {
	// init_batch fixed to 16
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/" // !! NEED to remove "_test"
	out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
	fc_out := 10    // 100 for cifar100
	init_pow := 6.0 // covers [-2^pow, 2^pow] values at ReLU evaluation
	mid_pow := 6.0
	final_pow := 6.0
	/*
	if cf100 {
		weight_dir = "Resnet_weights/weights_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		out_dir = "Resnet_enc_results/results_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		fc_out = 100 // 100 for cifar100
		if ker_wid == 3 {
			final_pow = 7.0
		} else if ker_wid == 5 {
			final_pow = 6.0
		} else {
			final_pow = 5.0
		}
		init_pow = 5.0
		mid_pow = 5.0
	}
	*/
	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth (not in 8, 14, 20)!")
	}
	real_batch := []int{16, 32, 64} // same as python (small for faster eval test) !! NEEDS to be changed for real test input {16, 32, 64}
	norm := []int{4, 8, 16}         // only use 1/norm batches among full batches (i.e., sparse packing)
	step := []int{1, 1, 1}          // non-one only when it is for inside

	logN := 16 // !! NEEDS to be modified to 16
	alpha := 0.0
	in_wids := []int{32, 16, 8}                                         // before cropping
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2} // same as python
	fast_pack := true
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}

	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, "Resnet_crop_sparse")
	//return 
	name := ""
	ct_input := make([]*ckks.Ciphertext, end)
	pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
	
	
	enc_start_xuchao := time.Now()
	ct_layer := ct_input
	pow := init_pow
	
	for iter := st; iter < end; iter++ {
		fmt.Println("Running ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(0)+".csv", in_wids[0]*in_wids[0]*3)
		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		input := make([]float64, cont.N)
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k] // sparse pack the input
						
					}
					k++
				}
			}
		}
		fmt.Println("Input: ")
		//prt_mat_norm(input, max_batch[0], norm[0], 3, false)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)
		enc_start := time.Now()
		//pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input[iter] = cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(enc_start))

		//timings := make([]float64, 6)
		//begin_start := time.Now()
		//sstart := time.Now()
	}
	
	//for i := 1; i <= 1; i++ {
	for i := 1; i <= num_blcs[0]; i++ {
	
		// ResNet Block 1
		
		//for i := 1; i <= num_blcs[0]; i++ {
		
		for iter := st; iter < end; iter++ {
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", real_batch[0])
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", real_batch[0])
			// bn_a := make([]float64, real_batch[0])
			// bn_b := make([]float64, real_batch[0])
			// for i := range bn_a {
			// 	bn_a[i] = 0.2
			// 	bn_b[i] = 0.0
			// }
			ker_in_batch := 3
			if i != 1 {
				ker_in_batch = real_batch[0]
			}
			ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", ker_in_batch*real_batch[0]*ker_size)
			// ker_in := make([]float64, ker_in_batch*real_batch[0]*ker_size)
			// for i := range ker_in {
			// 	ker_in[i] = 0.05 * float64(i) / float64(len(ker_in))
			// }
			
			name = "Block1_" + strconv.Itoa(i)
			
			ct_layer[iter] = evalConv_BN_multipic(cont, ct_layer[iter], ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, ker_in_batch, real_batch[0], norm[0], 0, step[0], 2, 2, "Conv_sparse", fast_pack, debug, name)
			
						
		}
		
		gap := 4
		fo := end / gap
		if end % gap > 0 {
			fo = fo + 1
		}
		res_ct := make([]*ckks.Ciphertext, fo)
		for it := 0; it < fo; it++ {
			aaa := 0
			
			if end - gap * it >= gap {
				aaa = gap
				
			} else {
				aaa = end - gap * it
				
			}
			res_ct[it] = megre_multipic(cont, it * gap, aaa, ct_layer)
				//cont.evaluator.Rotate(ct_layer[1], 1, ct_layer[1])
				//conv := cont.evaluator.AddNew(ct_layer[0], ct_layer[1])
			res_ct_tmp := make([]*ckks.Ciphertext, aaa)	
			res_ct_tmp = evalConv_Rule_multipic(cont, res_ct[it], alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, real_batch[0], norm[0], 0, step[0], 2, 0, aaa,"Conv_sparse", fast_pack, debug, name)
			
							
			for ita := 0; ita < aaa; ita++ {
				ct_layer[it * gap + ita] = res_ct_tmp[ita]		
			}
			
			ct_layer = move_multipic(cont, gap * it, aaa, ct_layer)
						
		
			
		}
		res := make([]float64, 65536)
				
		cont.decryptor.Decrypt(ct_layer[0], pl_input)
		res = cont.encoder.DecodeCoeffs(pl_input)
		
			
			
		pow = mid_pow
		fmt.Println("Block1, Layer iter", i, "done!")
		
		fmt.Println("Block1 done.") // !!!! HERE is DONE
		
	}
	
	
	
	ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
	bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
	bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])
	name = "Block1_to_2"
	
	for iter := st; iter < end; iter++ {
		
		ct_layer[iter] = evalConv_BNRelu_new_write(cont, ct_layer[iter], ker_in12, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1], norm[1], 0, step[1], 2, 1, "StrConv_sparse", fast_pack, debug, name)
	}
		
	res := make([]float64, 65536)
				
	cont.decryptor.Decrypt(ct_layer[0], pl_input)
	res = cont.encoder.DecodeCoeffs(pl_input)
		
	
	
	for i := 1; i <= num_blcs[1]; i++ {
		bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
		bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
		ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
		
		for iter := st; iter < end; iter++ {
			// ker_in := make([]float64, ker_in_batch*real_batch[0]*ker_size)
			// for i := range ker_in {
			// 	ker_in[i] = 0.05 * float64(i) / float64(len(ker_in))
			// }
			
			name = "Block2_" + strconv.Itoa(i)
			
			ct_layer[iter] = evalConv_BN_multipic(cont, ct_layer[iter], ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, 3, "Conv_sparse", fast_pack, debug, name)
									
		}
		
		gap := 8
		fo := end / gap
		if end % gap > 0 {
			fo = fo + 1
		}
		res_ct := make([]*ckks.Ciphertext, fo)
		for it := 0; it < fo; it++ {
			aaa := 0
			
			if end - gap * it >= gap {
				aaa = gap
				
			} else {
				aaa = end - gap * it
				
			}
			res_ct[it] = megre_multipic(cont, it * gap, aaa, ct_layer)
				//cont.evaluator.Rotate(ct_layer[1], 1, ct_layer[1])
				//conv := cont.evaluator.AddNew(ct_layer[0], ct_layer[1])
			res_ct_tmp := make([]*ckks.Ciphertext, aaa)	
			res_ct_tmp = evalConv_Rule_multipic(cont, res_ct[it], alpha, pow, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], norm[1], 0, step[1], 2, 0, aaa,"Conv_sparse", fast_pack, debug, name)
			
			for ita := 0; ita < aaa; ita++ {
				ct_layer[it * gap + ita] = res_ct_tmp[ita]		
			}
			
			ct_layer = move_multipic(cont, gap * it, aaa, ct_layer)
								
		}
		
		res := make([]float64, 65536)				
		cont.decryptor.Decrypt(ct_layer[0], pl_input)
		res = cont.encoder.DecodeCoeffs(pl_input)
		
						
	}
	
	
	fmt.Printf("xuchao num_blcs[1] done in %s \n", time.Since(enc_start_xuchao))

	ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
	bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
	bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])	
	name = "Block2_to_3"
	
	
	for iter := st; iter < end; iter++ {			
		
		ct_layer[iter] = evalConv_BNRelu_new_write(cont, ct_layer[iter], ker_in23, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[1], real_batch[2], norm[2], 0, step[2], 2, 2, "StrConv_sparse", fast_pack, debug, name)
		//ct_layer[iter] = evalConv_BN_multipic(cont, ct_layer[iter], ker_in23, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[1], real_batch[2], norm[2], 0, step[2], 2, 2, "StrConv_sparse", fast_pack, debug, name)
		
			
	}
		
	res = make([]float64, 65536)				
	cont.decryptor.Decrypt(ct_layer[0], pl_input)
	res = cont.encoder.DecodeCoeffs(pl_input)	
	

		
	for i := 1; i <= num_blcs[2]; i++ {
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
		ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)
		if i == num_blcs[2] {
				pow = final_pow
		}
		for iter := st; iter < end; iter++ {
			// ker_in := make([]float64, ker_in_batch*real_batch[0]*ker_size)
			// for i := range ker_in {
			// 	ker_in[i] = 0.05 * float64(i) / float64(len(ker_in))
			// }
			
			name = "Block3_" + strconv.Itoa(i)
			
			ct_layer[iter] = evalConv_BN_multipic(cont, ct_layer[iter], ker_in3, bn_a3, bn_b3, alpha, pow, in_wids[2], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 4, "Conv_sparse", fast_pack, debug, name)
									
		}
		
		gap := 16			
		
		fo := end / gap
		if end % gap > 0 {
			fo = fo + 1
		}
		res_ct := make([]*ckks.Ciphertext, fo)
		for it := 0; it < fo; it++ {
			aaa := 0
			
			if end - gap * it >= gap {
				aaa = gap
				
			} else {
				aaa = end - gap * it
				
			}
			res_ct[it] = megre_multipic(cont, it * 16, aaa, ct_layer)
				//cont.evaluator.Rotate(ct_layer[1], 1, ct_layer[1])
				//conv := cont.evaluator.AddNew(ct_layer[0], ct_layer[1])
			res_ct_tmp := make([]*ckks.Ciphertext, aaa)	
			res_ct_tmp = evalConv_Rule_multipic(cont, res_ct[it], alpha, pow, in_wids[2], raw_in_wids[2], ker_wid, real_batch[2], norm[2], 0, step[2], 2, 0, aaa,"Conv_sparse", fast_pack, debug, name)
			
			
			for ita := 0; ita < aaa; ita++ {
				ct_layer[it * gap + ita] = res_ct_tmp[ita]		
			}
			
			ct_layer = move_multipic(cont, gap * it, aaa, ct_layer)
								
		}
		
		res := make([]float64, 65536)				
		cont.decryptor.Decrypt(ct_layer[0], pl_input)
		res = cont.encoder.DecodeCoeffs(pl_input)
		
						
	}
	

	ker_inf_wid := raw_in_wids[2]
	if ker_inf_wid%2 == 0 {
		ker_inf_wid++
	}
	ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)
	ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
	for i := range ker_inf {
		for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
			ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
		}
	}
	bn_af := make([]float64, fc_out)
	for i := range bn_af {
		bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
	}
	bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			// bn_bf := make([]float64, fc_out)
			// for i := range bn_bf {
			// 	bn_bf[i] = 1 * float64(i)
			// }
	for iter := st; iter < end; iter++ {
		ct_layer[iter] = evalConv_BN(cont, ct_layer[iter], ker_inf_, bn_af, bn_bf, in_wids[2], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false)
	}
	
	
	
	fmt.Printf("xuchao Encryption done in %s \n", time.Since(enc_start_xuchao))

	for iter := st; iter < end; iter++ {
		cont.decryptor.Decrypt(ct_layer[iter], pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
			
		res_out := prt_mat_one_norm(res_tmp, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
		fmt.Println("\n result: ", res_out[:fc_out])
		writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])
	}

	
	

}


func testResNet_crop_sparse(st, end, ker_wid, depth int, debug, cf100 bool) {
	// init_batch fixed to 16
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/" // !! NEED to remove "_test"
	out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
	fc_out := 10    // 100 for cifar100
	init_pow := 6.0 // covers [-2^pow, 2^pow] values at ReLU evaluation
	mid_pow := 6.0
	final_pow := 6.0
	if cf100 {
		weight_dir = "Resnet_weights/weights_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		out_dir = "Resnet_enc_results/results_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		fc_out = 100 // 100 for cifar100
		if ker_wid == 3 {
			final_pow = 7.0
		} else if ker_wid == 5 {
			final_pow = 6.0
		} else {
			final_pow = 5.0
		}
		init_pow = 5.0
		mid_pow = 5.0
	}

	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth (not in 8, 14, 20)!")
	}
	real_batch := []int{16, 32, 64} // same as python (small for faster eval test) !! NEEDS to be changed for real test input {16, 32, 64}
	norm := []int{4, 8, 16}         // only use 1/norm batches among full batches (i.e., sparse packing)
	step := []int{1, 1, 1}          // non-one only when it is for inside

	logN := 16 // !! NEEDS to be modified to 16
	alpha := 0.0
	in_wids := []int{32, 16, 8}                                         // before cropping
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2} // same as python
	fast_pack := true
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}

	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, "Resnet_crop_sparse")
	//return 
	name := ""
	startxuchao := time.Now()
	for iter := st; iter < end; iter++ {
		fmt.Println("Running ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		// image := make([]float64, in_wids[0]*in_wids[0]*3)
		// for i := range image {
		// 	image[i] = 1.0 - 1.0*float64(i)/float64(len(image))
		// }
		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		input := make([]float64, cont.N)
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k] // sparse pack the input
						
					}
					k++
				}
			}
		}
		fmt.Println("Input: ")
		//prt_mat_norm(input, max_batch[0], norm[0], 3, false)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		enc_start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(enc_start))

		timings := make([]float64, 6)
		begin_start := time.Now()
		start := time.Now()

		// ResNet Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blcs[0]; i++ {
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", real_batch[0])
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", real_batch[0])
			// bn_a := make([]float64, real_batch[0])
			// bn_b := make([]float64, real_batch[0])
			// for i := range bn_a {
			// 	bn_a[i] = 0.2
			// 	bn_b[i] = 0.0
			// }
			ker_in_batch := 3
			if i != 1 {
				ker_in_batch = real_batch[0]
			}
			ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", ker_in_batch*real_batch[0]*ker_size)
			// ker_in := make([]float64, ker_in_batch*real_batch[0]*ker_size)
			// for i := range ker_in {
			// 	ker_in[i] = 0.05 * float64(i) / float64(len(ker_in))
			// }
			name = "Block1_" + strconv.Itoa(i)
			ct_layer = evalConv_BNRelu_new_write(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, ker_in_batch, real_batch[0], norm[0], 0, step[0], 2, 2, "Conv_sparse", fast_pack, debug, name)
			pow = mid_pow
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done.") // !!!! HERE is DONE
		timings[0] = time.Since(start).Seconds()
		start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])
		// ker_in12 := make([]float64, real_batch[0]*real_batch[1]*ker_size)
		// for i := range ker_in12 {
		// 	ker_in12[i] = 0.05 * float64(i) / float64(len(ker_in12))
		// }
		// bn_a := make([]float64, real_batch[1])
		// bn_b := make([]float64, real_batch[1])
		// for i := range bn_a {
		// 	bn_a[i] = 0.1
		// 	bn_b[i] = 0.0
		// }
		name = "Block1_to_2"
		ct_layer = evalConv_BNRelu_new_write(cont, ct_layer, ker_in12, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1], norm[1], 0, step[1], 2, 1, "StrConv_sparse", fast_pack, debug, name)
		fmt.Println("Block1 to 2 done!")
		timings[1] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 2
		for i := 1; i <= num_blcs[1]; i++ {
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
			// bn_a2 := make([]float64, real_batch[1])
			// bn_b2 := make([]float64, real_batch[1])
			// ker_in2 := make([]float64, real_batch[1]*real_batch[1]*ker_size)
			// for i := range bn_a2 {
			// 	bn_a2[i] = 0.1
			// 	bn_b2[i] = 0.0
			// }
			// for i := range ker_in2 {
			// 	ker_in2[i] = 0.05 * float64(i) / float64(len(ker_in2))
			// }
			name = "Block2_" + strconv.Itoa(i)
			ct_layer = evalConv_BNRelu_new_write(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, 3, "Conv_sparse", fast_pack, debug, name)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done.")
		timings[2] = time.Since(start).Seconds()
		start = time.Now()

		ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])
		// bn_a3 := make([]float64, real_batch[2])
		// bn_b3 := make([]float64, real_batch[2])
		// ker_in23 := make([]float64, real_batch[1]*real_batch[2]*ker_size)
		// for i := range bn_a3 {
		// 	bn_a3[i] = 0.1
		// 	bn_b3[i] = 0.0
		// }
		// for i := range ker_in23 {
		// 	ker_in23[i] = 0.05 * float64(i) / float64(len(ker_in23))
		// }
		name = "Block2_to_3"
		ct_layer = evalConv_BNRelu_new_write(cont, ct_layer, ker_in23, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[1], real_batch[2], norm[2], 0, step[2], 2, 2, "StrConv_sparse", fast_pack, debug, name)
		fmt.Println("Block2 to 3 done!")
		timings[3] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 3
		for i := 1; i <= num_blcs[2]; i++ {
			bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
			bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
			ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)
			// bn_a3 := make([]float64, real_batch[2])
			// bn_b3 := make([]float64, real_batch[2])
			// ker_in3 := make([]float64, real_batch[2]*real_batch[2]*ker_size)
			// for i := range bn_a3 {
			// 	bn_a3[i] = 0.1
			// 	bn_b3[i] = 0.0
			// }
			// for i := range ker_in3 {
			// 	ker_in3[i] = 0.1 * float64(i) / float64(len(ker_in3))
			// }

			if i == num_blcs[2] {
				pow = final_pow
			}
			name = "Block3_" + strconv.Itoa(i)
			ct_layer = evalConv_BNRelu_new_write(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, pow, in_wids[2], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 4, "Conv_sparse", fast_pack, debug, name)
			fmt.Println("Block3, Layer ", i, "done!")
		}
		fmt.Println("Block3 done.")
		
		timings[4] = time.Since(start).Seconds()
		start = time.Now()
		
		ker_inf_wid := raw_in_wids[2]
		if ker_inf_wid%2 == 0 {
			ker_inf_wid++
		}
		ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)
		// ker_inf := make([]float64, real_batch[2]*fc_out)
		// for i := range ker_inf {
		// 	ker_inf[i] = 0.1 * float64(i)
		// }
		var ct_result, ct_result2 *ckks.Ciphertext
		if cf100 {
			ker_inf_1 := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out/2)
			ker_inf_2 := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out/2)
			for i := 0; i < fc_out/2; i++ {
				for j := 0; j < real_batch[2]; j++ {
					for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
						ker_inf_1[j*fc_out/2+i+b*real_batch[2]*fc_out/2] = ker_inf[j*fc_out+i]
						ker_inf_2[j*fc_out/2+i+b*real_batch[2]*fc_out/2] = ker_inf[j*fc_out+i+fc_out/2]
					}
				}
			}
			bn_af := make([]float64, fc_out/2)
			for i := range bn_af {
				bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
			}
			bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			bn_bf_1 := make([]float64, fc_out/2)
			bn_bf_2 := make([]float64, fc_out/2)
			for i := range bn_bf_1 {
				bn_bf_1[i] = bn_bf[i]
				bn_bf_2[i] = bn_bf[i+fc_out/2]
			}
			ct_result = evalConv_BN(cont, ct_layer, ker_inf_1, bn_af, bn_bf_1, in_wids[2], ker_inf_wid, real_batch[2], fc_out/2, norm[2], float64(1<<30), false)
			ct_result2 = evalConv_BN(cont, ct_layer, ker_inf_2, bn_af, bn_bf_2, in_wids[2], ker_inf_wid, real_batch[2], fc_out/2, norm[2], float64(1<<30), false)
			fmt.Println("Final FC done.")
			timings[5] = time.Since(start).Seconds()
			start = time.Now()
		} else {
			ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
			for i := range ker_inf {
				for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
					ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
				}
			}
			bn_af := make([]float64, fc_out)
			for i := range bn_af {
				bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
			}
			bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			// bn_bf := make([]float64, fc_out)
			// for i := range bn_bf {
			// 	bn_bf[i] = 1 * float64(i)
			// }
			ct_result = evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[2], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false)
			fmt.Println("Final FC done.")
			timings[5] = time.Since(start).Seconds()
			start = time.Now()
		}

		fmt.Println()
		fmt.Println("===============  DECRYPTION  ===============")
		fmt.Println()
		if cf100 {
			cont.decryptor.Decrypt(ct_result, pl_input)
			res_tmp1 := cont.encoder.DecodeCoeffs(pl_input)
			cont.decryptor.Decrypt(ct_result2, pl_input)
			res_tmp2 := cont.encoder.DecodeCoeffs(pl_input)
			fmt.Printf("Decryption Done in %s \n", time.Since(start))
			res_out := append(prt_mat_one_norm(res_tmp1, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)[:fc_out/2], prt_mat_one_norm(res_tmp2, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)[:fc_out/2]...)
			fmt.Println("\n result: ", res_out)
			writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out)
		} else {
			cont.decryptor.Decrypt(ct_result, pl_input)
			res_tmp := cont.encoder.DecodeCoeffs(pl_input)
			fmt.Printf("Decryption Done in %s \n", time.Since(start))
			writeTxt("final_baseline",res_tmp)
			
			res_out := prt_mat_one_norm(res_tmp, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
			fmt.Println("\n result: ", res_out[:fc_out])
			writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])
		}

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Blc2->3: ", timings[3], " sec")
		fmt.Println("Blc3: ", timings[4], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
		
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
	fmt.Printf("Encryption done startxuchao all in %s \n", time.Since(startxuchao))

}

func testResNet_crop_fast_in(st, end, ker_wid, depth int, debug, cf100 bool) {
	// init_batch fixed to 16
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
	out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
	fc_out := 10    // 100 for cifar100
	init_pow := 6.0 // covers [-2^pow, 2^pow] values at ReLU evaluation
	mid_pow := 6.0
	final_pow := 6.0
	if cf100 {
		weight_dir = "Resnet_weights/weights_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		out_dir = "Resnet_enc_results/results_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid1/"
		fc_out = 100 // 100 for cifar100
		if ker_wid == 3 {
			final_pow = 7.0
		} else if ker_wid == 5 {
			final_pow = 6.0
		} else {
			final_pow = 5.0
		}
		init_pow = 5.0
		mid_pow = 5.0
	}

	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth (not in 8, 14, 20)!")
	}
	real_batch := []int{16, 32, 64} // same as python
	norm := []int{4, 2, 1}          // only use 1/norm batches
	step := []int{1, 2, 4}
	prt_start := []int{1, 1, 1}
	if ker_wid == 5 {
		prt_start[0] = 1
		prt_start[1] = 2
		prt_start[2] = 4
	}

	logN := 16
	alpha := 0.0
	in_wids := []int{32, 16, 8}                                         // before cropping
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2} // same as python
	fast_pack := true
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}

	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, "Resnet_crop_fast")

	for iter := st; iter < end; iter++ {
		fmt.Println("Running ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid1/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		input := make([]float64, cont.N)
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k]
					}
					k++
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat_norm(input, max_batch[0], norm[0], 1, false)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		enc_start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(enc_start))

		timings := make([]float64, 6)
		begin_start := time.Now()
		start := time.Now()

		// ResNet Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blcs[0]; i++ {
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", real_batch[0])
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", real_batch[0])
			ker_in_batch := 3
			if i != 1 {
				ker_in_batch = real_batch[0]
			}
			ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", ker_in_batch*real_batch[0]*ker_size)
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, ker_in_batch, real_batch[0], norm[0], 0, step[0], 2, 0, "Conv_inside", fast_pack, debug)
			pow = mid_pow
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done.")
		timings[0] = time.Since(start).Seconds()
		start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		ker_in12_new := make([]float64, 2*real_batch[0]*real_batch[1]*ker_size)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[0]; i++ {
				for j := 0; j < real_batch[1]; j++ {
					ker_in12_new[k*2*real_batch[0]*real_batch[1]+2*i*real_batch[1]+j] = ker_in12[k*real_batch[0]*real_batch[1]+i*real_batch[1]+j]
				}
			}
		}
		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in12_new, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, 0, "StrConv_inside", fast_pack, debug)
		fmt.Println("Block1 to 2 done!")
		if debug {
			max_bat := cont.N / (in_wids[0] * in_wids[0])
			res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_layer))
			prt_mat_norm_step(res_ttmp, max_bat, norm[1], step[1], prt_start[1], 3, false)
		}
		timings[1] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 2
		for i := 1; i <= num_blcs[1]; i++ {
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)

			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, 0, "Conv_inside", fast_pack, debug)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done.")
		timings[2] = time.Since(start).Seconds()
		start = time.Now()

		ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])
		ker_in23_new := make([]float64, 2*real_batch[1]*real_batch[2]*ker_size)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[1]; i++ {
				for j := 0; j < real_batch[2]; j++ {
					ker_in23_new[k*2*real_batch[1]*real_batch[2]+2*i*real_batch[2]+j] = ker_in23[k*real_batch[1]*real_batch[2]+i*real_batch[2]+j]
				}
			}
		}
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in23_new, bn_a3, bn_b3, alpha, pow, in_wids[0], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 0, "StrConv_inside", fast_pack, debug)
		fmt.Println("Block2 to 3 done!")
		if debug {
			max_bat := cont.N / (in_wids[0] * in_wids[0])
			res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_layer))
			prt_mat_norm_step(res_ttmp, max_bat, norm[2], step[2], prt_start[2], 3, false)
		}
		timings[3] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 3
		for i := 1; i <= num_blcs[2]; i++ {
			bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
			bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
			ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)

			if i == num_blcs[2] {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, pow, in_wids[0], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 0, "Conv_inside", fast_pack, debug)
			fmt.Println("Block3, Layer ", i, "done!")
		}
		fmt.Println("Block3 done.")
		timings[4] = time.Since(start).Seconds()
		start = time.Now()

		ker_inf_wid := raw_in_wids[0]
		if ker_inf_wid%2 == 0 {
			ker_inf_wid++
		}
		ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)
		var ct_result, ct_result2 *ckks.Ciphertext
		if cf100 {
			ker_inf_1 := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out/2)
			ker_inf_2 := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out/2)
			for i := 0; i < fc_out/2; i++ {
				for j := 0; j < real_batch[2]; j++ {
					for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
						ker_inf_1[j*fc_out/2+i+b*real_batch[2]*fc_out/2] = ker_inf[j*fc_out+i]
						ker_inf_2[j*fc_out/2+i+b*real_batch[2]*fc_out/2] = ker_inf[j*fc_out+i+fc_out/2]
					}
				}
			}
			bn_af := make([]float64, fc_out/2)
			for i := range bn_af {
				bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
			}
			bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			bn_bf_1 := make([]float64, fc_out/2)
			bn_bf_2 := make([]float64, fc_out/2)
			for i := range bn_bf_1 {
				bn_bf_1[i] = bn_bf[i]
				bn_bf_2[i] = bn_bf[i+fc_out/2]
			}
			ct_result = evalConv_BN(cont, ct_layer, ker_inf_1, bn_af, bn_bf_1, in_wids[0], ker_inf_wid, real_batch[2], fc_out/2, norm[2], float64(1<<30), false)
			ct_result2 = evalConv_BN(cont, ct_layer, ker_inf_2, bn_af, bn_bf_2, in_wids[0], ker_inf_wid, real_batch[2], fc_out/2, norm[2], float64(1<<30), false)
			fmt.Println("Final FC done.")
			timings[5] = time.Since(start).Seconds()
			start = time.Now()
		} else {
			ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
			for i := range ker_inf {
				for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
					ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
				}
			}
			bn_af := make([]float64, fc_out)
			for i := range bn_af {
				bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
			}
			bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
			ct_result = evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[0], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false)
			fmt.Println("Final FC done.")
			timings[5] = time.Since(start).Seconds()
			start = time.Now()
		}

		fmt.Println()
		fmt.Println("===============  DECRYPTION  ===============")
		fmt.Println()
		if cf100 {
			cont.decryptor.Decrypt(ct_result, pl_input)
			res_tmp1 := cont.encoder.DecodeCoeffs(pl_input)
			cont.decryptor.Decrypt(ct_result2, pl_input)
			res_tmp2 := cont.encoder.DecodeCoeffs(pl_input)
			fmt.Printf("Decryption Done in %s \n", time.Since(start))
			res_out := append(prt_mat_one_norm(res_tmp1, max_batch[0], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)[:fc_out/2], prt_mat_one_norm(res_tmp2, max_batch[0], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)[:fc_out/2]...)
			fmt.Println("\n result: ", res_out)
			writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out)
		} else {
			cont.decryptor.Decrypt(ct_result, pl_input)
			res_tmp := cont.encoder.DecodeCoeffs(pl_input)
			fmt.Printf("Decryption Done in %s \n", time.Since(start))
			res_out := prt_mat_one_norm(res_tmp, max_batch[0], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
			// fmt.Print(res_out)
			fmt.Println("\n result: ", res_out[:fc_out])
			writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])
		}

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Blc2->3: ", timings[3], " sec")
		fmt.Println("Blc3: ", timings[4], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}

func testResNet_crop_sparse_wide(st, end, ker_wid, depth, wide_case int, debug, cf100 bool) {
	// init_batch fixed to 16
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
	out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
	fc_out := 10

	init_pow := 5.0
	mid_pow := 5.0 // needs to be 5.0 in k3 d20 w3 for best performance
	final_pow := 5.0
	if ker_wid == 5 {
		init_pow = 6.0
		mid_pow = 6.0
		final_pow = 6.0
	}

	if cf100 {
		weight_dir = "Resnet_weights/weights_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
		out_dir = "Resnet_enc_results/results_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
		fc_out = 100
		final_pow = 7.0
		init_pow = 5.0
		mid_pow = 5.0
		if (ker_wid == 5) && (depth == 8) {
			init_pow = 6.0
			final_pow = 6.0
		}
	}

	init_batch := 16

	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth case (not in 8,14,20)!")
	}
	real_batch := []int{32, 64, 128} // same as python
	norm := []int{2, 4, 8}           // only use 1/norm batches
	log_sparse := []int{1, 2, 3}
	step := []int{1, 1, 1}
	kind := "Resnet_crop_sparse_wide2"

	if wide_case == 3 {
		real_batch = []int{48, 96, 192}
		norm = []int{1, 2, 4}
		log_sparse = []int{0, 1, 2}
		kind = "Resnet_crop_sparse_wide3"
	} else if wide_case != 2 {
		panic("wrong wide_case (2 nor 3)!")
	}

	logN := 16
	alpha := 0.0
	in_wids := []int{32, 16, 8}                                         // before cropping
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2} // same as python
	fast_pack := true
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}

	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, kind)

	for iter := st; iter < end; iter++ {
		fmt.Println("Running ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid"+strconv.Itoa(wide_case)+"/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)

		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid"+strconv.Itoa(wide_case)+"/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		input := make([]float64, cont.N)
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k]
					}
					k++
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat_norm(input, max_batch[0], norm[0], 3, false)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		enc_start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(enc_start))
		enc_start = time.Now()

		timings := make([]float64, 6)
		begin_start := time.Now()
		start := time.Now()

		// ResNet Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blcs[0]; i++ {
			if i == 5 {
				pow = mid_pow
			}
			var bn_batch int
			if i == 1 {
				bn_batch = init_batch
			} else {
				bn_batch = real_batch[0]
			}
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", bn_batch)
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", bn_batch)

			if i == 1 {
				ker_in := readTxt(weight_dir+"w0-conv.csv", 3*init_batch*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, 3, init_batch, norm[0], 0, step[0], 2, log_sparse[0], "Conv_sparse", fast_pack, debug)
				// pow = mid_pow
			} else if i == 2 {
				ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", init_batch*real_batch[0]*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, init_batch, real_batch[0], norm[0], 0, step[0], 2, log_sparse[0], "Conv_sparse", fast_pack, debug)
			} else {
				ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", real_batch[0]*real_batch[0]*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, real_batch[0], real_batch[0], norm[0], 0, step[0], 2, log_sparse[0], "Conv_sparse", fast_pack, debug)
			}
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done.")
		timings[0] = time.Since(start).Seconds()
		start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		ker_in12_0 := make([]float64, len(ker_in12)/2)
		ker_in12_1 := make([]float64, len(ker_in12)/2)
		if wide_case == 3 {
			for k := 0; k < ker_size; k++ {
				for i := 0; i < real_batch[0]; i++ {
					for j := 0; j < real_batch[1]/2; j++ {
						ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j)]   // [i][2*j]
						ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j+1)] // [i][2*j+1]
					}
				}
			}
		}

		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])

		if wide_case == 2 {
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in12, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1], norm[1], 0, step[1], 2, log_sparse[0]-1, "StrConv_sparse", fast_pack, debug)
		} else if wide_case == 3 {
			bn_a_0 := make([]float64, real_batch[1]/2)
			bn_a_1 := make([]float64, real_batch[1]/2)
			bn_b_0 := make([]float64, real_batch[1]/2)
			bn_b_1 := make([]float64, real_batch[1]/2)
			for i := range bn_b_0 {
				bn_a_0[i] = bn_a[2*i]
				bn_a_1[i] = bn_a[2*i+1]
				bn_b_0[i] = bn_b[2*i]
				bn_b_1[i] = bn_b[2*i+1]
			}
			ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a_0, bn_b_0, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1]/2, norm[0], 0, step[1], 2, 0, "StrConv_sparse_full", fast_pack, debug)
			ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a_1, bn_b_1, alpha, pow, in_wids[0], raw_in_wids[1], ker_wid, real_batch[0], real_batch[1]/2, norm[0], 0, step[1], 2, 0, "StrConv_sparse_full", fast_pack, debug)

			xi := make([]float64, cont.N)
			xi[2] = 1.0
			xi_plain := ckks.NewPlaintext(cont.params, ct_result2.Level(), 1.0)
			cont.encoder.EncodeCoeffs(xi, xi_plain)
			cont.encoder.ToNTT(xi_plain)
			ct_result2 = cont.evaluator.MulNew(ct_result2, xi_plain)
			ct_layer = cont.evaluator.AddNew(ct_result1, ct_result2)
		}
		fmt.Println("Block1 to 2 done!")
		timings[1] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 2
		for i := 1; i <= num_blcs[1]; i++ {
			if i == 5 {
				pow = init_pow
			}
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)

			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, log_sparse[1], "Conv_sparse", fast_pack, debug)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done.")
		timings[2] = time.Since(start).Seconds()
		start = time.Now()

		pow = mid_pow
		ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])

		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in23, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[1], real_batch[2], norm[2], 0, step[2], 2, log_sparse[1]-1, "StrConv_sparse", fast_pack, debug)
		fmt.Println("Block2 to 3 done!")
		timings[3] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 3
		for i := 1; i <= num_blcs[2]; i++ {
			if i == 3 {
				pow = init_pow
			}
			if i == 5 {
				pow = mid_pow
			}
			bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
			bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
			ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)

			if i == num_blcs[2] {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, pow, in_wids[2], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, log_sparse[2], "Conv_sparse", fast_pack, debug)
			fmt.Println("Block3, Layer ", i, "done!")
		}
		fmt.Println("Block3 done.")
		timings[4] = time.Since(start).Seconds()
		start = time.Now()

		ker_inf_wid := raw_in_wids[2]
		if ker_inf_wid%2 == 0 {
			ker_inf_wid++
		}
		ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)

		ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
		for i := range ker_inf {
			for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
				ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
			}
		}
		bn_af := make([]float64, fc_out)
		for i := range bn_af {
			bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
		}
		bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)

		ct_result := evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[2], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false)
		fmt.Println("Final FC done.")
		timings[5] = time.Since(start).Seconds()
		start = time.Now()

		fmt.Println()
		fmt.Println("===============  DECRYPTION  ===============")
		fmt.Println()
		cont.decryptor.Decrypt(ct_result, pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
		fmt.Printf("Decryption Done in %s \n", time.Since(start))
		res_out := prt_mat_one_norm(res_tmp, max_batch[2], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
		fmt.Println("\n result: ", res_out[:fc_out])
		writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Blc2->3: ", timings[3], " sec")
		fmt.Println("Blc3: ", timings[4], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}

func testResNet_crop_fast_wide_in(st, end, ker_wid, depth, wide_case int, debug, cf100 bool) {
	// init_batch fixed to 16
	ker_name := "ker" + strconv.Itoa(ker_wid)
	weight_dir := "Resnet_weights/weights_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
	out_dir := "Resnet_enc_results/results_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
	fc_out := 10 // 100 for cifar100

	init_pow := 5.0
	mid_pow := 5.0 // needs to be 5.0 in k3 d20 w3 for best performance
	final_pow := 5.0
	if ker_wid == 5 {
		init_pow = 6.0
		mid_pow = 6.0
		final_pow = 6.0
	}

	if cf100 {
		weight_dir = "Resnet_weights/weights_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
		out_dir = "Resnet_enc_results/results_cf100_crop_" + ker_name + "_d" + strconv.Itoa(depth) + "_wid" + strconv.Itoa(wide_case) + "/"
		fc_out = 100 // 100 for cifar100
		final_pow = 7.0
		init_pow = 5.0
		mid_pow = 5.0
		if (ker_wid == 5) && (depth == 8) {
			init_pow = 6.0
			final_pow = 6.0
		}
	}

	init_batch := 16 // needs to be modified to 16

	var num_blcs [3]int
	switch depth {
	case 20:
		num_blcs[0], num_blcs[1], num_blcs[2] = 7, 5, 5
	case 14:
		num_blcs[0], num_blcs[1], num_blcs[2] = 5, 3, 3
	case 8:
		num_blcs[0], num_blcs[1], num_blcs[2] = 3, 1, 1
	default:
		panic("wrong depth case (not in 8,14,20)!")
	}
	real_batch := []int{32, 64, 128} // same as python
	norm := []int{2, 4, 2}           // only use 1/norm batches
	step := []int{1, 1, 2}
	prt_start := []int{1, 1, 1}
	kind := "Resnet_crop_fast_wide2"
	if ker_wid == 5 {
		prt_start[0] = 1
		prt_start[1] = 1
		prt_start[2] = 2
	}
	if wide_case == 3 {
		real_batch = []int{48, 96, 192}
		norm = []int{1, 2, 1}
		kind = "Resnet_crop_fast_wide3"
	} else if wide_case != 2 {
		panic("wrong wide_case (2 nor 3)!")
	}

	logN := 16
	alpha := 0.0
	in_wids := []int{32, 16, 8}                                         // before cropping
	raw_in_wids := []int{32 - ker_wid/2, 16 - ker_wid/2, 8 - ker_wid/2} // same as python
	fast_pack := true
	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = (1 << logN) / (in_wids[i] * in_wids[i])
	}

	cont := newContext(logN, ker_wid, in_wids, raw_in_wids, true, kind)

	for iter := st; iter < end; iter++ {
		fmt.Println("Running ", iter, "-th iter... ker size: ", ker_wid)
		image := readTxt("Resnet_plain_data/crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid"+strconv.Itoa(wide_case)+"/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		if cf100 {
			image = readTxt("Resnet_plain_data/cf100_crop_ker"+strconv.Itoa(ker_wid)+"_d"+strconv.Itoa(depth)+"_wid"+strconv.Itoa(wide_case)+"/test_image_"+strconv.Itoa(iter)+".csv", in_wids[0]*in_wids[0]*3)
		}
		input := make([]float64, cont.N)
		k := 0
		for i := 0; i < in_wids[0]; i++ {
			for j := 0; j < in_wids[0]; j++ {
				for b := 0; b < 3; b++ {
					if (i < raw_in_wids[0]) && (j < raw_in_wids[0]) {
						input[i*in_wids[0]*max_batch[0]+j*max_batch[0]+b*norm[0]] = image[k]
					}
					k++
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat_norm(input, max_batch[0], norm[0], 1, false)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		enc_start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(enc_start))
		enc_start = time.Now()

		timings := make([]float64, 6)
		begin_start := time.Now()
		start := time.Now()

		// ResNet Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blcs[0]; i++ {
			if i == 5 {
				pow = mid_pow
			}
			var bn_batch int
			if i == 1 {
				bn_batch = init_batch
			} else {
				bn_batch = real_batch[0]
			}
			bn_a := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-a.csv", bn_batch)
			bn_b := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-b.csv", bn_batch)
			if i == 1 {
				ker_in := readTxt(weight_dir+"w0-conv.csv", 3*init_batch*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, 3, init_batch, norm[0], 0, step[0], 2, 0, "Conv", fast_pack, debug)
				// pow = mid_pow
			} else if i == 2 {
				ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", init_batch*real_batch[0]*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, init_batch, real_batch[0], norm[0], 0, step[0], 2, 0, "Conv", fast_pack, debug)
			} else {
				ker_in := readTxt(weight_dir+"w"+strconv.Itoa(i-1)+"-conv.csv", real_batch[0]*real_batch[0]*ker_size)
				ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in, bn_a, bn_b, alpha, pow, in_wids[0], raw_in_wids[0], ker_wid, real_batch[0], real_batch[0], norm[0], 0, step[0], 2, 0, "Conv", fast_pack, debug)
			}
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done.")
		timings[0] = time.Since(start).Seconds()
		start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		ker_in12_new := make([]float64, 2*real_batch[0]*real_batch[1]*ker_size)
		ker_in12_0 := make([]float64, len(ker_in12)/2)
		ker_in12_1 := make([]float64, len(ker_in12)/2)
		if wide_case == 2 {
			for k := 0; k < ker_size; k++ {
				for i := 0; i < real_batch[0]; i++ {
					for j := 0; j < real_batch[1]; j++ {
						ker_in12_new[k*2*real_batch[0]*real_batch[1]+2*i*real_batch[1]+j] = ker_in12[k*real_batch[0]*real_batch[1]+i*real_batch[1]+j]
					}
				}
			}
		} else if wide_case == 3 {
			for k := 0; k < ker_size; k++ {
				for i := 0; i < real_batch[0]; i++ {
					for j := 0; j < real_batch[1]/2; j++ {
						ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j)]   // [i][2*j]
						ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j+1)] // [i][2*j+1]
					}
				}
			}
		}

		bn_a := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-a.csv", real_batch[1])
		bn_b := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0])+"-b.csv", real_batch[1])

		if wide_case == 2 {
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in12_new, bn_a, bn_b, alpha, pow, in_wids[0], 2*raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[0]/2, 0, step[1], 2, 0, "StrConv_odd", fast_pack, debug)
		} else if wide_case == 3 {
			bn_a_0 := make([]float64, real_batch[1]/2)
			bn_a_1 := make([]float64, real_batch[1]/2)
			bn_b_0 := make([]float64, real_batch[1]/2)
			bn_b_1 := make([]float64, real_batch[1]/2)
			for i := range bn_b_0 {
				bn_a_0[i] = bn_a[2*i]
				bn_a_1[i] = bn_a[2*i+1]
				bn_b_0[i] = bn_b[2*i]
				bn_b_1[i] = bn_b[2*i+1]
			}
			ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a_0, bn_b_0, alpha, pow, in_wids[0], 2*raw_in_wids[1], ker_wid, real_batch[0], real_batch[0], norm[0], 0, step[1], 2, 0, "StrConv_odd", fast_pack, debug)
			ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a_1, bn_b_1, alpha, pow, in_wids[0], 2*raw_in_wids[1], ker_wid, real_batch[0], real_batch[0], norm[0], 2, step[1], 2, 0, "StrConv_odd", fast_pack, debug)
			ct_layer = cont.evaluator.AddNew(ct_result1, ct_result2)
		}
		fmt.Println("Block1 to 2 done!")
		if debug {
			max_bat := cont.N / (in_wids[1] * in_wids[1])
			res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_layer))
			prt_mat_norm_step(res_ttmp, max_bat, norm[1], step[1], prt_start[1], 3, false)
		}
		timings[1] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 2
		for i := 1; i <= num_blcs[1]; i++ {
			if i == 5 {
				pow = init_pow
			}
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-b.csv", real_batch[1])
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+i)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)

			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], raw_in_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, step[1], 2, 0, "Conv_inside", fast_pack, debug)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done.")
		timings[2] = time.Since(start).Seconds()
		start = time.Now()

		pow = mid_pow
		ker_in23 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-conv.csv", real_batch[1]*real_batch[2]*ker_size)
		bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-a.csv", real_batch[2])
		bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+1)+"-b.csv", real_batch[2])
		ker_in23_new := make([]float64, 2*real_batch[1]*real_batch[2]*ker_size)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[1]; i++ {
				for j := 0; j < real_batch[2]; j++ {
					ker_in23_new[k*2*real_batch[1]*real_batch[2]+2*i*real_batch[2]+j] = ker_in23[k*real_batch[1]*real_batch[2]+i*real_batch[2]+j]
				}
			}
		}
		ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in23_new, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 0, "StrConv_inside", fast_pack, debug)
		fmt.Println("Block2 to 3 done!")
		if debug {
			max_bat := cont.N / (in_wids[1] * in_wids[1])
			res_ttmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_layer))
			prt_mat_norm_step(res_ttmp, max_bat, norm[2], step[2], prt_start[2], 3, false)
		}
		timings[3] = time.Since(start).Seconds()
		start = time.Now()

		// ResNet Block 3
		for i := 1; i <= num_blcs[2]; i++ {
			if i == 3 {
				pow = init_pow
			}
			if i == 5 {
				pow = mid_pow
			}
			bn_a3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-a.csv", real_batch[2])
			bn_b3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-b.csv", real_batch[2])
			ker_in3 := readTxt(weight_dir+"w"+strconv.Itoa(num_blcs[0]+num_blcs[1]+i+1)+"-conv.csv", real_batch[2]*real_batch[2]*ker_size)

			if i == num_blcs[2] {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in3, bn_a3, bn_b3, alpha, pow, in_wids[1], raw_in_wids[2], ker_wid, real_batch[2], real_batch[2], norm[2], 0, step[2], 2, 0, "Conv_inside", fast_pack, debug)
			fmt.Println("Block3, Layer ", i, "done!")
		}
		fmt.Println("Block3 done.")
		timings[4] = time.Since(start).Seconds()
		start = time.Now()

		ker_inf_wid := raw_in_wids[1]
		if ker_inf_wid%2 == 0 {
			ker_inf_wid++
		}
		ker_inf := readTxt(weight_dir+"final-fckernel.csv", real_batch[2]*fc_out)
		ker_inf_ := make([]float64, ker_inf_wid*ker_inf_wid*real_batch[2]*fc_out)
		for i := range ker_inf {
			for b := 0; b < ker_inf_wid*ker_inf_wid; b++ {
				ker_inf_[i+b*real_batch[2]*fc_out] = ker_inf[i]
			}
		}
		bn_af := make([]float64, fc_out)
		for i := range bn_af {
			bn_af[i] = 1.0 / float64(raw_in_wids[2]*raw_in_wids[2]) // for reduce mean on raw_in_wids[2]**2 elements
		}
		bn_bf := readTxt(weight_dir+"final-fcbias.csv", fc_out)
		ct_result := evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[1], ker_inf_wid, real_batch[2], fc_out, norm[2], float64(1<<30), false)
		fmt.Println("Final FC done.")
		timings[5] = time.Since(start).Seconds()
		start = time.Now()

		fmt.Println()
		fmt.Println("===============  DECRYPTION  ===============")
		fmt.Println()
		cont.decryptor.Decrypt(ct_result, pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
		fmt.Printf("Decryption Done in %s \n", time.Since(start))
		res_out := prt_mat_one_norm(res_tmp, max_batch[1], norm[2], ker_inf_wid/2+1, ker_inf_wid/2+1)
		// fmt.Print(res_out)
		fmt.Println("\n result: ", res_out[:fc_out])
		writeTxt(out_dir+"class_result_"+ker_name+"_"+strconv.Itoa(iter)+".csv", res_out[:fc_out])

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Blc2->3: ", timings[3], " sec")
		fmt.Println("Blc3: ", timings[4], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[5], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}

func testImagenet_final_fast_in(st, end, ker_wid int) {
	// We use full packing: i.e., in_wid**2 element is contained in po2_in_wid**2 sized block <-> half padding of Resnet
	// So ReLU, keep or rot, StoC done on both the 1st & 2nd part of the CtoS ciphertexts
	ker_name := "ker" + strconv.Itoa(ker_wid) // "ker5"
	weight_dir := "weight_imgnet_" + ker_name + "_h5/"
	logN := 16
	raw_in_wids := []int{14, 7}   // same as python
	real_batch := []int{256, 512} // same as python
	iter := 2
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	var num_blc1, num_blc2 int
	if ker_name == "ker3" {
		in_wids[0] = 16
		in_wids[1] = 8
		kp_wids[0] = 14
		kp_wids[1] = 7
		num_blc1 = 3
		num_blc2 = 3
	} else if ker_name == "ker5" {
		in_wids[0] = 16
		in_wids[1] = 8
		kp_wids[0] = 14
		kp_wids[1] = 6
		num_blc1 = 3
		num_blc2 = 3
	} else {
		panic("strange ker name!")
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Imagenet_final_fast")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}
	alpha := 0.0 // 0.3 => leakyrelu
	init_pow := 6.0
	mid_pow := 5.0
	final_pow := 6.0

	// ker5_iter := []int{804, 886, 901, 956}
	// {3, 29, 87, 254, 357,
	// 399, 435, 455, 475, 476,
	// 518, 540, 545, 571, 631,
	// 657, 699, 711, 748, 790,
	// 804, 886, 901, 956}

	// for _, name_iter := range ker5_iter {
	for name_iter := st; name_iter < end; name_iter++ {
		weight_num := 10
		norm := 1
		fmt.Println("Start ", name_iter, "-th iter..")

		//raw_input := readTxt(ker_name+"_data/test_image_"+strconv.Itoa(name_iter)+".csv", raw_in_wids[0]*raw_in_wids[0]*real_batch[0])
		
		raw_input := readTxt("Resnet_plain_data/crop_ker3_d20_wid1/test_image_"+strconv.Itoa(name_iter)+".csv", raw_in_wids[0]*raw_in_wids[0]*real_batch[0])
		input := make([]float64, in_wids[0]*in_wids[0]*real_batch[0])
		for i := 0; i < raw_in_wids[0]; i++ {
			for j := 0; j < raw_in_wids[0]; j++ {
				for b := 0; b < real_batch[0]; b++ {
					input[i*in_wids[0]*real_batch[0]+j*real_batch[0]+b] = raw_input[i*raw_in_wids[0]*real_batch[0]+j*real_batch[0]+b]
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat(input, max_batch[0], 1)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(start))

		timings := make([]float64, 4)
		begin_start := time.Now()
		new_start := time.Now()

		// Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blc1; i++ {
			ker_in1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[0]*real_batch[0]*ker_size)
			weight_num++
			bn_a1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[0])
			bn_b1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[0])
			if i == num_blc1 {
				pow = mid_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in1, bn_a1, bn_b1, alpha, pow, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm, 0, 0, iter, 0, "Conv", false, false)
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done!")
		timings[0] = time.Since(new_start).Seconds()
		new_start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		weight_num++
		bn_a12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
		bn_b12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])

		ker_in12_0 := make([]float64, len(ker_in12)/2)
		ker_in12_1 := make([]float64, len(ker_in12)/2)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[0]; i++ {
				for j := 0; j < real_batch[1]/2; j++ {
					ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+j)]                 // [i][j]
					ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+real_batch[1]/2+j)] // [i][j+B/2]
				}
			}
		}
		bn_a12_0 := make([]float64, real_batch[1]/2)
		bn_a12_1 := make([]float64, real_batch[1]/2)
		bn_b12_0 := make([]float64, real_batch[1]/2)
		bn_b12_1 := make([]float64, real_batch[1]/2)
		for i := range bn_b12_0 {
			bn_a12_0[i] = bn_a12[i]
			bn_a12_1[i] = bn_a12[i+real_batch[1]/2]
			bn_b12_0[i] = bn_b12[i]
			bn_b12_1[i] = bn_b12[i+real_batch[1]/2]
		}

		// block1 to block 2
		ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a12_0, bn_b12_0, alpha, pow, in_wids[0], 2*kp_wids[1], ker_wid, real_batch[0], real_batch[0], norm, 0, 0, iter, 0, "StrConv", false, false)
		ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a12_1, bn_b12_1, alpha, pow, in_wids[0], 2*kp_wids[1], ker_wid, real_batch[0], real_batch[0], norm, 1, 0, iter, 0, "StrConv", false, false)
		ct_layer = cont.evaluator.AddNew(ct_result1, ct_result2)
		fmt.Println("Block1 to 2 done!")
		// res_tmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
		// prt_mat_norm(res_tmp, max_batch[1], 1, 4, false)
		timings[1] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// Block 2
		for i := 1; i <= num_blc2; i++ {
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
			weight_num++
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])
			if i == num_blc2 {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm, 0, 0, iter, 0, "Conv", false, false)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done!")
		timings[2] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// RMFC
		fin_out_batch := 1000
		ker_inf := readTxt(weight_dir+"fc.csv", real_batch[1]*fin_out_batch)
		bn_af := make([]float64, real_batch[1]*2)
		if ker_wid == 3 {
			for i := range bn_af {
				bn_af[i] = 1.0 / (7 * 7) // for reduce mean on 8*8 elements
			}
		} else {
			for i := range bn_af {
				bn_af[i] = 1.0 / (6 * 6) // for reduce mean on 8*8 elements
			}
		}
		bn_bf := make([]float64, real_batch[1]*2)
		for i := range bn_bf {
			bn_bf[i] = 0.0 //10.0 * float64(i)
		}
		ker_inf_ := make([]float64, 7*7*real_batch[1]*fin_out_batch)
		for b := 0; b < 7*7; b++ {
			for i := 0; i < real_batch[1]; i++ {
				for j := 0; j < fin_out_batch; j++ {
					ker_inf_[b*real_batch[1]*fin_out_batch+i*fin_out_batch+j] = ker_inf[i*fin_out_batch+j]
				}
			}
		}
		ct_result := evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[1], 7, real_batch[1], 1000, 1, float64(1<<30), false)
		timings[3] = time.Since(new_start).Seconds()
		new_start = time.Now()

		cont.decryptor.Decrypt(ct_result, pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
		fmt.Printf("Decryption done in %s \n", time.Since(new_start))
		final_result := prt_mat_one_norm(res_tmp, max_batch[1], 1, 4, 4)
		writeTxt(ker_name+"_enc_result/enc_result_"+strconv.Itoa(name_iter)+".csv", final_result[:1000])

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[3], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}

func testImagenet_sparse(st, end, ker_wid int) {
	// We use full packing: i.e., in_wid**2 element is contained in po2_in_wid**2 sized block <-> half padding of Resnet
	// So ReLU, keep or rot, StoC done on both the 1st & 2nd part of the CtoS ciphertexts
	debug := false
	ker_name := "ker" + strconv.Itoa(ker_wid) // "ker5"
	weight_dir := "weight_imgnet_" + ker_name + "_h5/"
	logN := 16
	raw_in_wids := []int{14, 7}   // same as python
	real_batch := []int{256, 512} // same as python
	log_sparse := []int{0, 1}
	norm := []int{1, 2}
	iter := 2
	in_wids := make([]int, len(raw_in_wids))
	kp_wids := make([]int, len(raw_in_wids))
	var num_blc1, num_blc2 int
	if ker_name == "ker3" {
		in_wids[0] = 16
		in_wids[1] = 8
		kp_wids[0] = 14
		kp_wids[1] = 7
		num_blc1 = 3
		num_blc2 = 3
	} else if ker_name == "ker5" {
		in_wids[0] = 16
		in_wids[1] = 8
		kp_wids[0] = 14
		kp_wids[1] = 6
		num_blc1 = 3
		num_blc2 = 3
	} else {
		panic("strange ker name!")
	}
	cont := newContext(logN, ker_wid, in_wids, kp_wids, true, "Imagenet_sparse")

	ker_size := ker_wid * ker_wid
	max_batch := make([]int, len(real_batch)) // the max batch
	for i := range max_batch {
		max_batch[i] = cont.N / (in_wids[i] * in_wids[i])
	}
	alpha := 0.0 // 0.3 => leakyrelu
	init_pow := 6.0
	mid_pow := 5.0
	final_pow := 6.0

	for name_iter := st; name_iter < end; name_iter++ {
		weight_num := 10
		fmt.Println("Start ", name_iter, "-th iter..")

		raw_input := readTxt(ker_name+"_data/test_image_"+strconv.Itoa(name_iter)+".csv", raw_in_wids[0]*raw_in_wids[0]*real_batch[0])
		input := make([]float64, in_wids[0]*in_wids[0]*real_batch[0])
		for i := 0; i < raw_in_wids[0]; i++ {
			for j := 0; j < raw_in_wids[0]; j++ {
				for b := 0; b < real_batch[0]; b++ {
					input[i*in_wids[0]*real_batch[0]+j*real_batch[0]+b] = raw_input[i*raw_in_wids[0]*real_batch[0]+j*real_batch[0]+b]
				}
			}
		}
		fmt.Println("Input: ")
		prt_mat(input, max_batch[0], 1)
		fmt.Println("vec size: ", cont.N)
		fmt.Println("input width: ", raw_in_wids)
		fmt.Println("kernel width: ", ker_wid)
		fmt.Println("num batches: ", real_batch)

		start := time.Now()
		pl_input := ckks.NewPlaintext(cont.params, cont.ECD_LV, cont.params.Scale()) // contain plaintext values
		cont.encoder.EncodeCoeffs(input, pl_input)
		ct_input := cont.encryptor.EncryptNew(pl_input)
		fmt.Printf("Encryption done in %s \n", time.Since(start))

		timings := make([]float64, 4)
		begin_start := time.Now()
		new_start := time.Now()

		// Block 1
		pow := init_pow
		ct_layer := ct_input
		for i := 1; i <= num_blc1; i++ {
			ker_in1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[0]*real_batch[0]*ker_size)
			weight_num++
			bn_a1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[0])
			bn_b1 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[0])
			if i == num_blc1 {
				pow = mid_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in1, bn_a1, bn_b1, alpha, pow, in_wids[0], kp_wids[0], ker_wid, real_batch[0], real_batch[0], norm[0], 0, 1, iter, log_sparse[0], "Conv_sparse", true, debug)
			fmt.Println("Block1, Layer ", i, "done!")
		}
		fmt.Println("Block1 done!")
		timings[0] = time.Since(new_start).Seconds()
		new_start = time.Now()

		ker_in12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[0]*real_batch[1]*ker_size)
		weight_num++
		bn_a12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
		bn_b12 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])

		ker_in12_0 := make([]float64, len(ker_in12)/2)
		ker_in12_1 := make([]float64, len(ker_in12)/2)
		for k := 0; k < ker_size; k++ {
			for i := 0; i < real_batch[0]; i++ {
				for j := 0; j < real_batch[1]/2; j++ {
					// ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+j)]                 // [i][j]
					// ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+real_batch[1]/2+j)] // [i][j+B/2]
					ker_in12_0[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j)]   // [i][2*j]
					ker_in12_1[k*real_batch[0]*real_batch[1]/2+(i*real_batch[1]/2+j)] = ker_in12[k*real_batch[0]*real_batch[1]+(i*real_batch[1]+2*j+1)] // [i][2*j+1]

				}
			}
		}
		bn_a12_0 := make([]float64, real_batch[1]/2)
		bn_a12_1 := make([]float64, real_batch[1]/2)
		bn_b12_0 := make([]float64, real_batch[1]/2)
		bn_b12_1 := make([]float64, real_batch[1]/2)
		for i := range bn_b12_0 {
			// bn_a12_0[i] = bn_a12[i]
			// bn_a12_1[i] = bn_a12[i+real_batch[1]/2]
			// bn_b12_0[i] = bn_b12[i]
			// bn_b12_1[i] = bn_b12[i+real_batch[1]/2]
			bn_a12_0[i] = bn_a12[2*i]
			bn_a12_1[i] = bn_a12[2*i+1]
			bn_b12_0[i] = bn_b12[2*i]
			bn_b12_1[i] = bn_b12[2*i+1]
		}

		// block1 to block 2
		// ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a12_0, bn_b12_0, alpha, pow, in_wids[0], 2*kp_wids[1], ker_wid, real_batch[0], real_batch[0], norm, 0, 0, iter, 0, "StrConv", false, false)
		// ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a12_1, bn_b12_1, alpha, pow, in_wids[0], 2*kp_wids[1], ker_wid, real_batch[0], real_batch[0], norm, 1, 0, iter, 0, "StrConv", false, false)
		// ct_layer = cont.evaluator.AddNew(ct_result1, ct_result2)

		ct_result1 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_0, bn_a12_0, bn_b12_0, alpha, pow, in_wids[0], kp_wids[1], ker_wid, real_batch[0], real_batch[1]/2, norm[0], 0, 1, 2, 0, "StrConv_sparse_full", true, debug)
		ct_result2 := evalConv_BNRelu_new(cont, ct_layer, ker_in12_1, bn_a12_1, bn_b12_1, alpha, pow, in_wids[0], kp_wids[1], ker_wid, real_batch[0], real_batch[1]/2, norm[0], 0, 1, 2, 0, "StrConv_sparse_full", true, debug)

		xi := make([]float64, cont.N)
		xi[2] = 1.0
		xi_plain := ckks.NewPlaintext(cont.params, ct_result2.Level(), 1.0)
		cont.encoder.EncodeCoeffs(xi, xi_plain)
		cont.encoder.ToNTT(xi_plain)
		ct_result2 = cont.evaluator.MulNew(ct_result2, xi_plain)
		ct_layer = cont.evaluator.AddNew(ct_result1, ct_result2)

		fmt.Println("Block1 to 2 done!")
		// res_tmp := cont.encoder.DecodeCoeffs(cont.decryptor.DecryptNew(ct_result))
		// prt_mat_norm(res_tmp, max_batch[1], 1, 4, false)
		timings[1] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// Block 2
		for i := 1; i <= num_blc2; i++ {
			ker_in2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-conv.csv", real_batch[1]*real_batch[1]*ker_size)
			weight_num++
			bn_a2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-a.csv", real_batch[1])
			bn_b2 := readTxt(weight_dir+"w"+strconv.Itoa(weight_num)+"-b.csv", real_batch[1])
			if i == num_blc2 {
				pow = final_pow
			}
			ct_layer = evalConv_BNRelu_new(cont, ct_layer, ker_in2, bn_a2, bn_b2, alpha, pow, in_wids[1], kp_wids[1], ker_wid, real_batch[1], real_batch[1], norm[1], 0, 1, iter, log_sparse[1], "Conv_sparse", true, debug)
			fmt.Println("Block2, Layer ", i, "done!")
		}
		fmt.Println("Block2 done!")
		timings[2] = time.Since(new_start).Seconds()
		new_start = time.Now()

		// RMFC
		fin_out_batch := 1000
		ker_inf := readTxt(weight_dir+"fc.csv", real_batch[1]*fin_out_batch)
		bn_af := make([]float64, real_batch[1]*2)
		if ker_wid == 3 {
			for i := range bn_af {
				bn_af[i] = 1.0 / (7 * 7) // for reduce mean on 8*8 elements
			}
		} else {
			for i := range bn_af {
				bn_af[i] = 1.0 / (6 * 6) // for reduce mean on 8*8 elements
			}
		}
		bn_bf := make([]float64, real_batch[1]*2)
		for i := range bn_bf {
			bn_bf[i] = 0.0 //10.0 * float64(i)
		}
		ker_inf_ := make([]float64, 7*7*real_batch[1]*fin_out_batch)
		for b := 0; b < 7*7; b++ {
			for i := 0; i < real_batch[1]; i++ {
				for j := 0; j < fin_out_batch; j++ {
					ker_inf_[b*real_batch[1]*fin_out_batch+i*fin_out_batch+j] = ker_inf[i*fin_out_batch+j]
				}
			}
		}
		ct_result := evalConv_BN(cont, ct_layer, ker_inf_, bn_af, bn_bf, in_wids[1], 7, real_batch[1], fin_out_batch, 1, float64(1<<30), false)
		fmt.Println("Final FC done.")
		timings[3] = time.Since(new_start).Seconds()
		new_start = time.Now()

		cont.decryptor.Decrypt(ct_result, pl_input)
		res_tmp := cont.encoder.DecodeCoeffs(pl_input)
		fmt.Printf("Decryption done in %s \n", time.Since(new_start))
		final_result := prt_mat_one_norm(res_tmp, max_batch[1], 1, 4, 4)
		writeTxt(ker_name+"_enc_result/enc_result_"+strconv.Itoa(name_iter)+".csv", final_result[:1000])

		fmt.Println("Blc1: ", timings[0], " sec")
		fmt.Println("Blc1->2: ", timings[1], " sec")
		fmt.Println("Blc2: ", timings[2], " sec")
		fmt.Println("Final (reduce_mean & FC): ", timings[3], " sec")
		fmt.Printf("Total done in %s \n", time.Since(begin_start))
	}
}
