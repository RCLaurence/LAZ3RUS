import re
import vtk

def get_weld_actor_from_dat(fname):
    '''
    Returns an actor list of sphere sources representing each of the positions from a KUKA dat file.
    The 'strike' block is red, the rest of the positions are gold.
    '''
    
    #vtk constants
    start_color = tuple(vtk.vtkNamedColors().GetColor3d("tomato"))
    main_color = tuple(vtk.vtkNamedColors().GetColor3d("wheat"))
    actor_collection = vtk.vtkActorCollection()
    
    flag = []
    with open(fname, 'r') as f:
        contents = f.readlines()
        line_no = 0
        for line in contents:
            if line.strip():
                if "WDAT" in line.strip():
                    # print(line_no, line.strip())
                    flag.append(line_no)
            line_no += 1
        #positions will be 2 lines after "WDAT", use regex to find decimal numbers
        
        for line_number in flag:
            pattern = r'\b\d*\.\d+\b|\b\d+\.\d*\b'
            position = [float(x) for x in re.findall(pattern, contents[line_number+2])]
            sphere = vtk.vtkSphereSource()
            sphere.SetPhiResolution(24)
            sphere.SetThetaResolution(24)
            sphere.SetRadius(2)
            sphere.SetCenter(tuple(position[:3]))
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            this_actor = vtk.vtkActor()
            this_actor.SetMapper(mapper)
            if actor_collection.GetNumberOfItems() == 0:
                this_actor.GetProperty().SetColor(start_color)
                c_actor = gen_caption_actor('S', this_actor, start_color)
            else:
                this_actor.GetProperty().SetColor(main_color)
            actor_collection.AddItem(this_actor)
    return actor_collection, c_actor
    
def gen_caption_actor(message, actor = None, color = (0,0,0)):
    '''
    Captions an actor
    '''
    caption_actor = vtk.vtkCaptionActor2D()
    if actor is not None:
        b = actor.GetBounds()
        caption_actor.SetAttachmentPoint((b[0],b[2],b[4]))
    caption_actor.SetCaption(message)
    caption_actor.SetThreeDimensionalLeader(False)
    caption_actor.BorderOff()
    caption_actor.LeaderOff()
    caption_actor.SetWidth(0.25 / 3.0)
    caption_actor.SetHeight(0.10 / 3.0)
    
    p = caption_actor.GetCaptionTextProperty()
    p.SetColor(color)
    p.BoldOn()
    p.ItalicOff()
    p.SetFontSize(16)
    p.ShadowOn()
    return caption_actor

if __name__ == "__main__":
    fname = 'Data/fix_W7_X_0_15_50.dat'
    actors, c_actor = get_weld_actor_from_dat(fname)

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    actors.InitTraversal()
    for i in range(actors.GetNumberOfItems()):
        ren.AddActor(actors.GetNextActor())
    ren.AddActor(c_actor)
    iren.Initialize()
    ren.ResetCamera()
    renWin.Render()
    iren.Start()