


import sys
import pandas as pd



class args_obj:
    def __init__(self):
        txt = ""
        print_schema = False
        pass

def parse_args(args):
    index = 0
    ret = args_obj()
    while(index < len(args)):
        print(args[index])
        if args[index] == "-txt":
            #the next vals is filepath to the txt
            ret.txt = args[index+1]
            index += 1
        elif args[index] == "-schema":
            args_obj.print_schema = True;
        index += 1


def main():
    parse_args(sys.argv)
    
    pass



if __name__ =="__main__":
    main()