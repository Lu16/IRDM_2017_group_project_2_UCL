package data_pre_processing;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

public class obtain_feature_value {

	public static void main(String[] args) {
		try {
			
//			File file = new File("D:\\DM_sub_data\\Folder5\\vali_feature.txt");
			File file = new File("D:\\DM_sub_data\\Folder5\\test_feature.txt");
			
//			FileReader fr = new FileReader("D:\\DM_sub_data\\Folder5\\vali.txt");
			FileReader fr = new FileReader("D:\\MSLR\\Fold5\\test.txt");
			
			BufferedReader br = new BufferedReader(fr);
			FileWriter fw = new FileWriter(file);
			BufferedWriter bw = new BufferedWriter(fw);
			
			
			
			while(true){
				
				String line = br.readLine();
				if(line==null){
					break;
				}
				String[] elements_with_colon = line.split(" ");
				System.out.println(elements_with_colon.length);
				
				String write_line = "";
				for(int j=0; j<elements_with_colon.length; j++){
					
					if (j<1){
						write_line = write_line + elements_with_colon[j] + " ";
					}
					else{
						String[] elements_without_colon = elements_with_colon[j].split(":");
						write_line = write_line + elements_without_colon[1] + " ";
					}

				}
				bw.write(write_line.trim());
				bw.newLine();

				
			}
			
			bw.flush();
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

}
