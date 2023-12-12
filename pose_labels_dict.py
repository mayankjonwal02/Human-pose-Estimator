LabelDict = {
    0:"Random",
    1:"Thank You",
    2:"Peace",
    3:"Good Luck",
    4:"Hello",
    5:"Break",
    6:"Nice",
    7:"Fuck You",
    8:"Call Me",
    9:"Bad Luck",
    10:"Washroom",
    11:"Shoulder",
    12:"Back",
    13:"Come here"
    
}



class PoseData:
    def __init__(self, *args, **kwargs):
        self.task = "fetch data"

    def fetch_labels(self):
        return LabelDict



hello = PoseData()
print(hello.fetch_labels())