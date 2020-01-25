#!/bin/bash

# Number of images that should be used
NDATASETS=1

#SPOKES=41
SPOKES=31 #less information -> more artifacts 
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
mkdir -p "./Images_Z"

for ((p_num = 1; p_num <= $NDATASETS; p_num++))
{	
	# Beginning of per image create_sliced
	echo -----------------------------------------------------------------
	echo "Starting to work on picture "$p_num
	filename="./P"$p_num"/kspace"
	outputfilename="./P"$p_num"/output"$p_num
    mkdir -p "./P"$p_num"/slices"
    bart fft -i 7 $filename $outputfilename	

    #create poisson undersampling pattern
    accel_y=1.5
    accel_z=1.5
    mask_os="./P"$p_num"/poisson_mask"
    outputfilename_undersampled="./P"$p_num"/output_undersampled"$p_num
    bart poisson -Y $(bart show -d1 $filename) -Z $(bart show -d2 $filename) -y $accel_y -z $accel_z -C 32 -v -e $mask_os 
    bart fmac $filename $mask_os tmp1
    bart fft -i 7 tmp1 $outputfilename_undersampled

    #comment out to use dimensions at top 
    NX=$(bart show -d0 $filename)
    NY=$(bart show -d1 $filename)
    NZ=$(bart show -d2 $filename)
	
	# Beginning of per image header of create_undersample
	filename="./P"$p_num"/kspace"
	mkdir -p "./P"$p_num"/reco_slice"
	
	# Only for testing.
	#NX=0
	#NY=0
	#NZ=36
	
    #create undersampled k_space 
    for(( ix = 0; ix < $NX; ix++))
	{
	
	#Create the 2d Slice
	echo "creating image sliced through dimension x with slice number: "$ix

        #input files names
	slice_input="./P"$p_num"/slices/slicex"$ix
        #slice_input_rss="./P"$p_num"/slices/slicex_rss"$ix
        slice_input_rss="./Images_X/P"$p_num"_slicex"$ix
        slice_input_png="./Images_X/P"$p_num"_slicex"$ix".png"

        #input slices
	bart slice 0 $ix $outputfilename $slice_input 
        bart rss $(bart bitmask 3) $slice_input $slice_input_rss
        bart toimg $slice_input_rss $slice_input_png 

        #output file names
        slice_output_und="./P"$p_num"/slices/slicex_undersampled"$ix
        #slice_output_und_rss="./P"$p_num"/slices/slicex_undersampled_rss"$ix
        slice_output_und_rss="./Images_X/P"$p_num"_undersampled_slicex"$ix
        slice_output_png="./Images_X/P"$p_num"_undersampled_slicex"$ix".png"
        
        #slices etc
        bart slice 0 $ix $outputfilename_undersampled $slice_output_und
        bart rss $(bart bitmask 3) $slice_output_und $slice_output_und_rss
        bart toimg $slice_output_und_rss $slice_output_png
		
	# Remove the two 2d slices
        #rm ""$slice_output_und".hdr"
        #rm ""$slice_output_und".cfl"
        #rm ""$slice_input".hdr"
        #rm ""$slice_input".cfl"
		
	}
	for(( iy = 50; iy < $NY; iy++))
    {
	# Create the 2d Slice #
	echo "creating image sliced through dimension y with slice number: "$iz

        #input files names
	slice_input="./P"$p_num"/slices/slicey"$iy
        #slice_input_rss="./P"$p_num"/slices/slicey_rss"$iy
        slice_input_rss="./Images_Y/P"$p_num"_slicey"$iy
        slice_input_png="./Images_Y/P"$p_num"_slicey"$iy".png"

        #input slices
	bart slice 1 $iy $outputfilename $slice_input 
        bart rss $(bart bitmask 3) $slice_input $slice_input_rss
        bart toimg $slice_input_rss $slice_input_png 


        #output file names
        slice_output_und="./P"$p_num"/slices/slice_y_undersampled"$iy
        #slice_output_und_rss="./P"$p_num"/slices/slice_y_undersampled_rss"$iy
        slice_output_und_rss="./Images_Y/P"$p_num"_reco_slicey"$iy
        slice_output_png="./Images_Y/P"$p_num"_reco_slicey"$iy".png"
        
        #slices etc
        bart slice 1 $iy $outputfilename_undersampled $slice_output_und
        bart rss $(bart bitmask 3) $slice_output_und $slice_output_und_rss
        bart toimg $slice_output_und_rss $slice_output_png
		
		# Remove the two 2d slices
        #rm ""$slice_output_und".hdr"
        #rm ""$slice_output_und".cfl"
        #rm ""$slice_input".hdr"
        #rm ""$slice_input".cfl"
		
	}
	for(( iz = 35; iz < $NZ; iz++))
	{	
		# Create the 2d Slice #
		echo "creating image sliced through dimension z with slice number: "$iz

		#input files names
		slice_input="./P"$p_num"/slices/slicez"$iz
		#slice_input_rss="./P"$p_num"/slices/slicez_rss"$iz
		slice_input_rss="./Images_Z/P"$p_num"_slicez"$iz
		slice_input_png="./Images_Z/P"$p_num"_slicez"$iz".png"

		#input slices
			bart slice 2 $iz $outputfilename $slice_input 
		bart rss $(bart bitmask 3) slice_input slice_input_rss
		bart toimg slice_input_rss $slice_input_png 


		#output file names
		slice_output_und="./P"$p_num"/slices/slicez_undersampled"$iz
		#slice_output_und_rss="./P"$p_num"/slices/slicez_undersampled_rss"$iz
		slice_output_und_rss="./Images_Z/P"$p_num"_reco_slicez"$iz
		slice_output_png="./Images_Z/P"$p_num"_reco_slicez"$iz".png"

		#slices etcsaved_net_images_knees_smallnsaved_net_images_knees_smallnet_20_0.13et_20_0.13
		bart slice 2 $iz $outputfilename_undersampled $slice_output_und
		bart rss $(bart bitmask 3) $slice_output_und $slice_output_und_rss
		bart toimg $slice_output_und_rss $slice_output_png
			
			# Remove the two 2d slices
		#rm ""$slice_output_und".hdr"
		#rm ""$slice_output_und".cfl"
		#rm ""$slice_input".hdr"
		#rm ""$slice_input".cfl"
   }
   echo -----------------------------------------------------------------
   echo "Finished with Picture "$p_num 
   
   # delete the output file
   rm ""$outputfilename".hdr"
   rm ""$outputfilename".cfl"
   rm ""$outputfilename_undersampled".cfl"
   rm ""$outputfilename_undersampled".hdr"
   rm ""$mask_os".cfl"
   rm ""$mask_os".hdr"
}
rm tmp1* tmp2* 
