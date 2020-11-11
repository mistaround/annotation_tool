import argparse
import Annotation

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog="Annotation Tool")
    parser.add_argument('-i','--INFILE',help='Input image',required=True)
    parser.add_argument('-m','--MASK',help='Mask image',required=True)
    parser.add_argument('-o','--OUTFILE',help='Output New Mask',required=True)

    args = parser.parse_args()
    App = Annotation.Annotation(args.INFILE, args.MASK, args.OUTFILE)
    App.run()
    