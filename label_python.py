import pathlib

from astropy.io import fits

HEAD_DIR = Path(os.environ["THESIS"])
MODSPEC_PATH = HEAD_DIR / "Modspec" / "Modspec_Flatproc"

nightfolders = MODSPEC_PATH.glob("night*")

for nightpath in nightfolders:
    with open(MODSPEC_PATH / "{0}_Kepler.txt".format(
                nightpath.name.capitalize()), "w") as keplerobjs, 
    open( 
                    MODSPEC_PATH / "{0}_Kepler_Arc.txt".format(
                        nightpath.name.capitalize()), "w") as keplerarcs,
    open(
        MODSPEC_PATH / "{0}_Standard.txt".format(
            nightpath.name.capitalize()), "w")
    standardarcs = open(
        MODSPEC_PATH / "{0}_Standard_Arc.txt".format(
            nightpath.name.capitalize()), "w")
    nightfiles = nightpath.glob("{0}.*.fit".format(nightpath.name))

    for nfile in nightfiles:

