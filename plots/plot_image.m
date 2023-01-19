function res = plot_image(filename)

    img = imread(filename);
    imshow(img);
    res = 1;

end