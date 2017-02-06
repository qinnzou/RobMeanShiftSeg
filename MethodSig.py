
def RGB2LUV(RGB)
    return LUV

def CovTr(LUVImg)
    return TraceOfCov

def GetR(TraceOfCov, UserInput_Segmentation Level)
    return R

//Main Loop
def Rand25(ImgW, ImgH, PresentMask)
    Avoid Locations which are masked out
    return Locations25(x,y)

def 25x9PPtsL(Locations(x,y), LUVImg)
    return 25x9PointsLUV, LUVCenters

def Mapp(25x9PointsLUV, Locations(x,y), LUVImg, R)
    Lookup Vectors LUVImg[Locations]
    Compute MaxCenter
    return MaxCenterIdx

def MeanShift(iteration_diff = 0.1)
    // Look out for extreme saddle points
    return FeaturePointsUsedToComputeLastMean, Mode

def UpdateRemMask(PresentMask, FeaturePointsUsedToComputeLastMean)
    Remove Pixels from image and LUV domain
    Remove 8Conn Pixels from Image domain
    return UpdatedRemMask, PixelToModeAssocMask

def MainIter()
    Load Nmin, Ncon, R as categorical values
    do Until, NumberOfPts from MeanShift < Nmin
    All above functions from "//Main Loop"
    return SetOfAllModes(Cx,Cy,ModeNo)

def PruneModes(SetOfAllModes, Nmin, Ncon, PixelToModeAssocMask)
    return PrunedModes, UnallocatedPixelMask

def InflateModesAndReallocate(R, PrunedModes, UnallocatedPixleMask, PixelToModeAssocMask)
    SingleWindow Candidate - Just allocate to that
    Multiple Possible Candidates, do next
        New Pixels are added to this color, if it has at least one neighbor in image domain allocated to this color
    return UpdatedAllocPixelMask

def PostProcessing(UpdatedAllocatedMask, Ncon)
    return FinalSegmentedMask
