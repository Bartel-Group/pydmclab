USE_INSTALLED_PYDMC = True

if USE_INSTALLED_PYDMC:
    from pydmc.CompTools import CompTools
    from pydmc.MagTools import MagTools
    
else:
    from CompTools import CompTools
    from MagTools import MagTools
    
def main():
    return

if __name__ == '__main__':
    main()