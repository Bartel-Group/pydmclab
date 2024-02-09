# from pydmclab.utils.handy import read_json, write_json
from jobflow import SETTINGS


class Store(object):
    def __init__(
        self, formula=None, ID=None, standard=None, mag=None, xc=None, calc=None
    ):
        store = SETTINGS.JOB_STORE
        store.connect()

        self.store = store
        # self.key = "--".join([formula, ID, standard, mag, xc])
        self.formula = formula
        self.ID = ID
        self.standard = standard
        self.mag = mag
        self.xc_calc = "-".join([xc, calc]) if xc and calc else None

    @property
    def query(self):
        # need to specify which job to grab..
        # formula, ID, standard, mag, xc = self.key.split("--")
        criteria = {}
        if self.formula:
            criteria["output.formula"] = self.formula
        if self.ID:
            criteria["output.ID"] = self.ID
        if self.standard:
            criteria["output.standard"] = self.standard
        if self.mag:
            criteria["output.mag"] = self.mag
        if self.xc_calc:
            criteria["output.xc_calc"] = self.xc_calc

        return self.store.query(criteria=criteria)

    @property
    def results(self):
        query = self.query
        return [r for r in query]

    @property
    def outputs(self):
        results = self.results
        keys = []
        outputs = []
        for r in results:
            output = r["output"]
            outputs.append(output)
            labels = ["formula", "ID", "standard", "mag", "xc_calc"]
            stuff = []
            for l in labels:
                stuff.append(output[l] if l in output else None)

            key = "--".join([str(v) for v in stuff])
            keys.append(key)
        return dict(zip(keys, outputs))


d = Store().outputs
