#!/bin/bash

# initialize bart
export OMP_NUM_THREADS=8
export TOOLBOX_PATH=/home/tim/bart-0.4.04
export PATH=${TOOLBOX_PATH}:${PATH}

# Number of images that should be used
NDATASETS=20
NDATA_START=1

#SPOKES=41
SPOKES=5 #less information -> more artifacts 
TURNS=5

#generate radial trajectory for undersampling
traj_os="traj_os"
traj_os_t=""$traj_os"_t"
bart traj -r -x$((128*2)) -y$(($SPOKES*$TURNS)) -t$TURNS tmp1  
bart scale 0.5 tmp1 $traj_os
bart reshape $(bart bitmask 2 10) $SPOKES $TURNS $traj_os $traj_os_t

#create directory for images if they don't exist yet.
mkdir -p "./Images_X"
mkdir -p "./Images_Y"

for ((p_num=$NDATA_START; p_num <= $NDATASETS; p_num++))
{	
	# Beginning of per image create_sliced
	echo -----------------------------------------------------------------
	echo "Starting to work on picture "$p_num
	filename="./P"$p_num"/kspace"
	outputfilename="./P"$p_num"/output"$p_num
	mkdir -p "./P"$p_num"/slices"
	bart fft -i 7 $filename $outputfilename	

	#create poisson undersampling pattern
	accel_y=1.9
	accel_z=1.9


	mask_os="./P"$p_num"/poisson_mask"
	outputfilename_undersampled="./P"$p_num"/output_undersampled"$p_num
	bart poisson -Y $(bart show -d1 $filename) -Z $(bart show -d2 $filename) -y $accel_y -z $accel_z -C 32 -v -e $mask_os 
	bart fmac $filename $mask_os tmp1
	bart fft -i 7 tmp1 $outputfilename_undersampled

	#comment out to use dimensions at top 
	#NX=$(bart show -d0 $filename)
	#NY=$(bart show -d1 $filename)
	#NZ=$(bart show -d2 $filename)
	echo x: $(bart show -d0 $filename), y: $(bart show -d1 $filename), z :$(bart show -d2 $filename)
	
	# Beginning of per image header of create_undersample
	filename="./P"$p_num"/kspace"
	mkdir -p "./P"$p_num"/reco_slice"

	# Define starting and end points for all pictures.
	start_x=15
	stop_x=300	
	start_y=50
	stop_y=280
	start_z=40
	stop_z=220
	
    for(( ix=$start_x; ix < $stop_x; ix++))
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
		slice_output_und_rss="./Images_Y/P"$p_num"_undersampled_slicex"$ix
		slice_output_png="./Images_Y/P"$p_num"_u_slicex"$ix".png"

		#slices etc
		bart slice 0 $ix $outputfilename_undersampled $slice_output_und
		bart rss $(bart bitmask 3) $slice_output_und $slice_output_und_rss
		bart toimg $slice_output_und_rss $slice_output_png

		# Remove the two 2d slices
		rm ""$slice_output_und".hdr"
		rm ""$slice_output_und".cfl"
		rm ""$slice_input".hdr"
		rm ""$slice_input".cfl"
		rm ""$slice_input_rss".hdr"
		rm ""$slice_input_rss".cfl"
		rm ""$slice_output_und_rss".hdr"
		rm ""$slice_output_und_rss".cfl"

		
	}
	for(( iy=$start_y; iy < $stop_y; iy++))
	{	
		# Create the 2d Slice #
		echo "creating image sliced through dimension y with slice number: "$iy

		#input files names
		slice_input="./P"$p_num"/slices/slicey"$iy
		#slice_input_rss="./P"$p_num"/slices/slicey_rss"$iy
		slice_input_rss="./Images_X/P"$p_num"_slicey"$iy
		slice_input_png="./Images_X/P"$p_num"_slicey"$iy".png"

		#input slices
		bart slice 1 $iy $outputfilename $slice_input 
		bart rss $(bart bitmask 3) $slice_input $slice_input_rss
		bart toimg $slice_input_rss $slice_input_png 


		#output file names
		slice_output_und="./P"$p_num"/slices/slice_y_undersampled"$iy
		slice_output_und_rss="./Images_Y/P"$p_num"_reco_slicey"$iy
		slice_output_png="./Images_Y/P"$p_num"_u_slicey"$iy".png"
		
		#slices etc
		bart slice 1 $iy $outputfilename_undersampled $slice_output_und
		bart rss $(bart bitmask 3) $slice_output_und $slice_output_und_rss
		bart toimg $slice_output_und_rss $slice_output_png
			
		# Remove the two 2d slices
		rm ""$slice_output_und".hdr"
		rm ""$slice_output_und".cfl"
		rm ""$slice_input".hdr"
		rm ""$slice_input".cfl"
		rm ""$slice_input_rss".hdr"
		rm ""$slice_input_rss".cfl"
		rm ""$slice_output_und_rss".hdr"
		rm ""$slice_output_und_rss".cfl"
		
	}
	for(( iz=$start_z; iz < $stop_z; iz++))
	{	
		# Create the 2d Slice #
		echo "creating image sliced through dimension z with slice number: "$iz

		#input files names
		slice_input="./P"$p_num"/slices/slicez"$iz
		#slice_input_rss="./P"$p_num"/slices/slicez_rss"$iz
		slice_input_rss="./Images_X/P"$p_num"_slicez"$iz
		slice_input_png="./Images_X/P"$p_num"_slicez"$iz".png"


		#input slices
		bart slice 2 $iz $outputfilename $slice_input 
		bart rss $(bart bitmask 3) $slice_input $slice_input_rss
		bart toimg $slice_input_rss $slice_input_png 


		#output file names
		slice_output_und="./P"$p_num"/slices/slicez_undersampled"$iz
		#slice_output_und_rss="./P"$p_num"/slices/slicez_undersampled_rss"$iz
		slice_output_und_rss="./Images_Y/P"$p_num"_reco_slicez"$iz
		slice_output_png="./Images_Y/P"$p_num"_u_slicez"$iz".png"

		
		#slices etcsaved_net_images_knees_smallnsaved_net_images_knees_smallnet_20_0.13et_20_0.13
		bart slice 2 $iz $outputfilename_undersampled $slice_output_und
		bart rss $(bart bitmask 3) $slice_output_und $slice_output_und_rss
		bart toimg $slice_output_und_rss $slice_output_png
			
		# Remove the two 2d slices
		rm ""$slice_output_und".hdr"
		rm ""$slice_output_und".cfl"
		rm ""$slice_input".hdr"
		rm ""$slice_input".cfl"
		rm ""$slice_input_rss".hdr"
		rm ""$slice_input_rss".cfl"
		rm ""$slice_output_und_rss".hdr"
		rm ""$slice_output_und_rss".cfl"
   }
   echo -----------------------------------------------------------------
   echo "Finished with Picture "$p_num 
   
   # delete the output file
   rm ""$outputfilename".hdr"
   rm ""$outputfilename".cfl"
   rm ""$outputfilename_undersampled".hdr"
   rm ""$outputfilename_undersampled".cfl"

}
#rm tmp1* tmp2* 
