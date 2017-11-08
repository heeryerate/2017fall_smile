import glob, os

for c, i in enumerate(glob.glob('final data/*')):
  print(' * converting', i)
  os.system('convert ' + i + ' ' + i.replace('.TIFF','.jpg'))