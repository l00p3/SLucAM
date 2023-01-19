function img_filename= load_img_filename(image_name_filename)
    
    f = fopen(image_name_filename);
    img_filename = fgetl(f);
    fclose(f);

end