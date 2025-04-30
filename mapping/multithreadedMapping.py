import threading
import os
import time




def generateMap():
	img_source_path = "C:\\Users\\Reece\\datasets\\dataset2\\images"
	work_path = "C:\\Users\\Reece\\datasets\\odm_data_aukerman111"
	result_path = "C:\\Users\\Reece\\datasets\\odm_data_aukerman111\\multiThreadedMapping"
	group_size = 50
	processed_imgs = []
	orthophoto_number = 0

	while(1):
		img_names = os.listdir(img_source_path)
		img_names = sorted(img_names)
		print(img_names)

		copied_imgs = []
		
		#Copy source images into work folder if they have not already been processed. Do as many as many as group_size at once if they are available
		for img in img_names:
			if(len(copied_imgs) >= group_size):
				break

			if(not(img in processed_imgs)):
				print("copy " + img_source_path + "\\" + img + " " + work_path + "\\images")
				os.system("copy " + img_source_path + "\\" + img + " " + work_path + "\\images")
				processed_imgs.append(img)
				copied_imgs.append(img)

		#Process images in work folder
		os.system('docker run -ti --rm --memory 6GB -v C:/Users/Reece/datasets:/datasets opendronemap/odm --project-path /datasets odm_data_aukerman111 --feature-quality low --feature-type orb --fast-orthophoto --rerun-all --pc-quality low')
		
		#Remove images from work folder
		for img in copied_imgs:
			print(work_path + "\\images\\" + img)
			os.remove(work_path + "\\images\\" + img)

		#Copy orthophoto to result path after it is available
		#print("Waiting for orthophoto")
		#print("odm_orthophoto.original.tiff in")
		#print(os.listdir(work_path + "\\odm_orthophoto"))
		#while(not("odm_orthophoto.original.tif" in os.listdir(work_path + "\\odm_orthophoto"))):pass
		print("copy " + work_path + "\\odm_orthophoto\\odm_orthophoto.original.tif " + result_path + "\\orthophoto" + str(orthophoto_number) + ".tif")
		os.system("copy " + work_path + "\\odm_orthophoto\\odm_orthophoto.original.tif " + result_path + "\\orthophoto" + str(orthophoto_number) + ".tif")	
		orthophoto_number = orthophoto_number + 1
#"C:\\Users\\Reece\\datasets\\odm_data_aukerman111\\odm_orthophoto\\odm_orthophoto.original.tiff" 
		
					

mappingThread = threading.Thread(target=generateMap)
mappingThread.start()



while(1):
	time.sleep(1)
	print("Waiting...")
	
