package data_pre_processing;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

public class split_data {

	public static void main(String[] args) {
		try {
			
			FileReader fr = new FileReader("D:\\DM_sub_data\\Folder1\\vali.txt");
			
			BufferedReader br = new BufferedReader(fr);
			
			int count = 0;
			while(true){
				String line = br.readLine();
				count = count + 1;
				if(line==null){
					break;
				}
				
			}
			br.close();
			fr.close();
			
			File file = new File("D:\\DM_sub_data\\Folder1\\vali_sub.txt");
			
			FileReader fr2 = new FileReader("D:\\DM_sub_data\\Folder1\\vali.txt");
			
			BufferedReader br2 = new BufferedReader(fr2);
			FileWriter fw = new FileWriter(file);
			BufferedWriter bw = new BufferedWriter(fw);
			
			int write_count = (int)count/4;
			
			for(int i=0; i<write_count; i++){
				String line = br2.readLine();
	
				bw.write(line);
				bw.newLine();
			}
			
			bw.flush();
			bw.close();
			
			br2.close();
			fr2.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
