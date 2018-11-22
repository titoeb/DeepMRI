#!/bin/bash

# Number of images that should be used
NDATASETS=1

SPOKES=41
TURNS=5

#generate radial trajectory for undersampling
traj_os="traj_os"
traj_os_t=""$traj_os"_t"
bart traj -r -x$((128*2)) -y$(($SPOKES*$TURNS)) -t$TURNS tmp1  
bart scale 0.5 tmp1 $traj_os
bart reshape $(bart bitmask 2 10) $SPOKES $TURNS $traj_os $traj_os_t

#create not existing directory for images
mkdir -p "./Images_X"
mkdir -p "./Images_Y"

for ((p_num = 1; p_num <= $NDATASETS; p_num++))
{	
	# Beginning of per image create_sliced
	echo -----------------------------------------------------------------
	echo "Starting to work on picture "$p_num
	filename="./P"$p_num"/kspace"
	outputfilename="./P"$p_num"/output"$p_num
    mkdir -p "./P"$p_num"/slices"
    bart fft -i 7 $filename $outputfilename	

    #comment out to use dimensions at top 
    NX=$(bart show -d0 $filename)
    NY=$(bart show -d1 $filename)
    NZ=$(bart show -d2 $filename)
	
	# Beginning of per image header of create_undersample
	filename="./P"$p_num"/kspace"
	mkdir -p "./P"$p_num"/reco_slice"
	
	# Only for testing.
	NX=0
	NY=0
	#NZ=36
	
    for(( ix = 0; ix < $NX; ix++))
	{	
		# Create the 2d Slice #
		echo "creating image sliced through dimension x with slice number: "$ix
		slice_output_1="./P"$p_num"/slices/slicex"$ix
		bart slice 0 $ix $outputfilename $slice_output_1
		
		# Create image and artefact image
		echo "reconstructing image sliced through dimension x with slice number: "$ix
		slice_input="./P"$p_num"/slices/slicex"$ix
		slice_input_png="./Images_X/P"$p_num"_slicex"$ix".png"
        slice_output="./P"$p_num"/reco_slice/reco_slicex"$ix
        slice_output_png="./Images_Y/P"$p_num"_reco_slicex"$ix".png"
        bart transpose 0 2 $slice_input tmp1
        bart rss $(bart bitmask 3) tmp1 tmp2
        x_res=$(bart show -d 0 tmp1)
        y_res=$(bart show -d 1 tmp1)
        bart toimg tmp2 $slice_input_png 

		bart nufft $traj_os tmp1 tmp2 
        bart transpose 2 3 tmp2 tmp1
        bart reshape $(bart bitmask 3 10) $SPOKES $TURNS tmp1 tmp2
        bart transpose 2 3 tmp2 tmp1
        bart nufft -i -d$x_res:$y_res:1 $traj_os_t tmp1 tmp2
        bart rss $(bart bitmask 3) tmp2 $slice_output
        
        bart toimg $slice_output $slice_output_png
		
		# Remove the two 2d slices
		rm ""$slice_output_1".hdr"
		rm ""$slice_output_1".cfl"
		rm ""$slice_output".hdr"
		rm ""$slice_output".cfl"
		
	}
	for(( iy = 50; iy < $NY; iy++))
	{	
		# Create the 2d Slice #
		echo "creating image sliced through dimension y with slice number: "$iy
		slice_output_1="./P"$p_num"/slices/slicey"$iy
		bart slice 1 $iy $outputfilename $slice_output_1
		
		# Create image and artefact image
        echo "reconstructing image sliced through dimension y with slice number: "$iy
		slice_input="./P"$p_num"/slices/slicey"$iy
		slice_input_png="./Images_X/P_"$p_num"_slicey"$iy".png"
        slice_output="./P"$p_num"/reco_slice/reco_slicey"$iy
        slice_output_png="./Images_Y/P_"$p_num"_reco_slicey"$iy".png"
        bart transpose 1 2 $slice_input tmp1
        bart rss $(bart bitmask 3) tmp1 tmp2
        x_res=$(bart show -d 0 tmp1)
        y_res=$(bart show -d 1 tmp1)
        bart toimg tmp2 $slice_input_png 

		bart nufft $traj_os tmp1 tmp2 
        bart transpose 2 3 tmp2 tmp1
        bart reshape $(bart bitmask 3 10) $SPOKES $TURNS tmp1 tmp2
        bart transpose 2 3 tmp2 tmp1
        bart nufft -i -d$x_res:$y_res:1 $traj_os_t tmp1 tmp2
        bart rss $(bart bitmask 3) tmp2 $slice_output
        
        bart toimg $slice_output $slice_output_png
		
		# Remove the two 2d slices
		rm ""$slice_output_1".hdr"
		rm ""$slice_output_1".cfl"
		rm ""$slice_output".hdr"
		rm ""$slice_output".cfl"
		
	}
	for(( iz = 35; iz < $NZ; iz++))
	{	
	 	# Create the 2d Slice
		echo "creating image sliced through dimension z with slice number: "$iz
		slice_output_1="./P"$p_num"/slices/slicez"$iz
		bart slice 2 $iz $outputfilename $slice_output_1
		
		# Create image and artefact image
        echo "reconstructing image sliced through dimension z with slice number: "$iz
		
		slice_input="./P"$p_num"/slices/slicez"$iz
		slice_input_png="./Images_X/P_"$p_num"_slicez"$iz".png"
        slice_output="./P"$p_num"/reco_slice/reco_slicez"$iz
        slice_output_png="./Images_Y/P_"$p_num"_reco_slicez"$iz".png"
        bart transpose 0 0 $slice_input tmp1
        bart rss $(bart bitmask 3) tmp1 tmp2
        x_res=$(bart show -d 0 tmp1)
        y_res=$(bart show -d 1 tmp1)
        bart toimg tmp2 $slice_input_png 

		bart nufft $traj_os tmp1 tmp2 
        bart transpose 2 3 tmp2 tmp1
        bart reshape $(bart bitmask 3 10) $SPOKES $TURNS tmp1 tmp2
        bart transpose 2 3 tmp2 tmp1
        bart nufft -i -d$x_res:$y_res:1 $traj_os_t tmp1 tmp2
        bart rss $(bart bitmask 3) tmp2 $slice_output
        
        bart toimg $slice_output $slice_output_png	
		
		# Remove the two 2d slices
		rm ""$slice_output_1".hdr"
		rm ""$slice_output_1".cfl"
		rm ""$slice_output".hdr"
		rm ""$slice_output".cfl"
   }
   echo -----------------------------------------------------------------
   echo "Finished with Picture "$p_num 
   
   # delete the output file
   rm ""$outputfilename".hdr"
   rm ""$outputfilename".cfl"
}
rm tmp1* tmp2* 
