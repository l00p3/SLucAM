ffmpeg -framerate $1 -i images/image_%03d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p videos/output.mp4
