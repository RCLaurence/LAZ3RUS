import FreeCAD, Part, Sketcher
import Mesh
import ObjectsFem
import Fem

App.newDocument()
App.getDocument('Unnamed').addObject('PartDesign::Body','Body')
App.getDocument('Unnamed').getObject('Body').newObject('Sketcher::SketchObject','Sketch')
App.getDocument('Unnamed').getObject('Sketch').AttachmentSupport = (App.getDocument('Unnamed').getObject('XZ_Plane'),[''])
App.getDocument('Unnamed').getObject('Sketch').MapMode = 'FlatFace'


p = Part.Parabola(App.Vector(0,0,0),App.Vector(0,2.336748,0),App.Vector(0,0,1))
inter_x = 4.103842
inter_y = 0.534936
App.getDocument('Unnamed').getObject('Sketch').addGeometry(Part.ArcOfParabola(p,-inter_x,inter_x),False)

geoList = []
geoList.append(Part.LineSegment(App.Vector(inter_x, inter_y, 0.000000),App.Vector(-inter_x, inter_y, 0.000000)))
App.getDocument('Unnamed').getObject('Sketch').addGeometry(geoList,False)
del geoList    

constraintList = []
constraintList.append(Sketcher.Constraint('Coincident', 1, 1, 0, 2))
constraintList.append(Sketcher.Constraint('Coincident', 1, 2, 0, 1))
constraintList.append(Sketcher.Constraint('Horizontal', 1))
App.getDocument('Unnamed').getObject('Sketch').addConstraint(constraintList)
del constraintList

App.getDocument('Unnamed').getObject('Body').newObject('PartDesign::Revolution','Revolution')
App.getDocument('Unnamed').getObject('Revolution').Profile = (App.getDocument('Unnamed').getObject('Sketch'), ['',])
App.getDocument('Unnamed').getObject('Revolution').Angle = 90.000000
App.getDocument('Unnamed').getObject('Revolution').ReferenceAxis = (App.getDocument('Unnamed').getObject('Sketch'), ['Edge2'])


App.getDocument('Unnamed').getObject('Body').newObject('PartDesign::Pad','Pad')
App.getDocument('Unnamed').getObject('Pad').Profile = (App.getDocument('Unnamed').getObject('Revolution'), ['Face2',])
App.getDocument('Unnamed').getObject('Pad').Length = 101.820616

App.getDocument('Unnamed').getObject('Body').newObject('PartDesign::Revolution','Revolution001')
App.getDocument('Unnamed').getObject('Revolution001').Profile = (App.getDocument('Unnamed').getObject('Pad'), ['Face3',])
App.getDocument('Unnamed').getObject('Revolution001').Angle = 90.000000
App.getDocument('Unnamed').getObject('Revolution001').ReferenceAxis = (App.getDocument('Unnamed').getObject('Pad'), ['Edge8'])
App.getDocument('Unnamed').getObject('Revolution001').Reversed = 1

App.getDocument('Unnamed').recompute()    

App.getDocument('Unnamed').getObject('Body').newObject('PartDesign::Plane','DatumPlane')
App.getDocument('Unnamed').getObject('DatumPlane').AttachmentOffset = App.Placement(App.Vector(0.0000000000, 0.0000000000, 0.0000000000),  App.Rotation(0.0000000000, 0.0000000000, 0.0000000000))
App.getDocument('Unnamed').getObject('DatumPlane').MapReversed = False
App.getDocument('Unnamed').getObject('DatumPlane').AttachmentSupport = [(App.getDocument('Unnamed').getObject('Revolution001'),'Face4')]
App.getDocument('Unnamed').getObject('DatumPlane').MapPathParameter = 0.000000
App.getDocument('Unnamed').getObject('DatumPlane').MapMode = 'FlatFace'

App.getDocument('Unnamed').getObject('Body').newObject('Sketcher::SketchObject','Sketch001')
App.getDocument('Unnamed').getObject('Sketch001').AttachmentSupport = (App.getDocument('Unnamed').getObject('DatumPlane'),[''])
App.getDocument('Unnamed').getObject('Sketch001').MapMode = 'FlatFace'

ActiveSketch = App.getDocument('Unnamed').getObject('Sketch001')

lastGeoId = len(ActiveSketch.Geometry)

geoList = []
points = [App.Vector(-34.103842, 111.820616, 0),
App.Vector(-34.103842, -10.000000, 0),
App.Vector(34.103842, -10.000000, 0),
App.Vector(34.103842, 111.820616, 0),
]
geoList.append(Part.LineSegment(points[0], points[1]))
geoList.append(Part.LineSegment(points[1], points[2]))
geoList.append(Part.LineSegment(points[2], points[3]))
geoList.append(Part.LineSegment(points[3], points[0]))

App.getDocument('Unnamed').getObject('Sketch001').addGeometry(geoList,False)
del geoList

constraintList = []
constraintList.append(Sketcher.Constraint('Coincident', 0, 2, 1, 1))
constraintList.append(Sketcher.Constraint('Coincident', 1, 2, 2, 1))
constraintList.append(Sketcher.Constraint('Coincident', 2, 2, 3, 1))
constraintList.append(Sketcher.Constraint('Coincident', 3, 2, 0, 1))
constraintList.append(Sketcher.Constraint('Vertical', 0))
constraintList.append(Sketcher.Constraint('Vertical', 2))
constraintList.append(Sketcher.Constraint('Horizontal', 1))
constraintList.append(Sketcher.Constraint('Horizontal', 3))
App.getDocument('Unnamed').getObject('Sketch001').addConstraint(constraintList)
del constraintList

App.getDocument('Unnamed').getObject('Body').newObject('PartDesign::Pad','Pad001')
App.getDocument('Unnamed').getObject('Pad001').Profile = (App.getDocument('Unnamed').getObject('Sketch001'), ['',])
App.getDocument('Unnamed').getObject('Pad001').Length = 1.000000
App.getDocument('Unnamed').getObject('Pad001').TaperAngle = 0.000000
App.getDocument('Unnamed').getObject('Pad001').UseCustomVector = 0
App.getDocument('Unnamed').getObject('Pad001').Direction = (0, 0, -1)
App.getDocument('Unnamed').getObject('Pad001').ReferenceAxis = (App.getDocument('Unnamed').getObject('Sketch001'), ['N_Axis'])
App.getDocument('Unnamed').getObject('Pad001').AlongSketchNormal = 1
App.getDocument('Unnamed').getObject('Pad001').Type = 0
App.getDocument('Unnamed').getObject('Pad001').UpToFace = None
App.getDocument('Unnamed').getObject('Pad001').Reversed = 0
App.getDocument('Unnamed').getObject('Pad001').Midplane = 0
App.getDocument('Unnamed').getObject('Pad001').Offset = 0

App.getDocument('Unnamed').recompute()
a = App.Placement(App.Matrix(-3.289325e-01,-9.443403e-01, 4.991505e-03, 7.605741e+02, 9.443529e-01,
 -3.289221e-01, 2.895909e-03,-1.004194e+03,-1.092921e-03, 5.665734e-03,
  9.999830e-01, 2.446998e+02, 0.000000e+00, 0.000000e+00, 0.000000e+00,
  1.000000e+00))
App.ActiveDocument.Body.Placement = a

__objs__ = []
__objs__.append(FreeCAD.getDocument("Unnamed").getObject("Body"))
Mesh.export(__objs__, u"C:/Work/The University of Manchester Dropbox/Matt Roy/Robin_Laurence/LAZ3RUS/Data/show.stl")
Part.export(__objs__, u"C:/Work/The University of Manchester Dropbox/Matt Roy/Robin_Laurence/LAZ3RUS/Data/show.step")

### Begin command FEM_MeshNetgenFromShape
ObjectsFem.makeMeshNetgenLegacy(FreeCAD.ActiveDocument, 'FEMMeshNetgen')
FreeCAD.ActiveDocument.ActiveObject.Shape = FreeCAD.ActiveDocument.Body
FreeCAD.ActiveDocument.ActiveObject.Fineness = 'Moderate'
FreeCAD.getDocument('Unnamed').getObject('FEMMeshNetgen').MaxSize = 0.5

FreeCAD.getDocument('Unnamed').getObject('FEMMeshNetgen').MinSize = 0.1

### End command FEM_MeshNetgenFromShape

App.getDocument('Unnamed').recompute()
# Gui.getDocument('Unnamed').resetEdit()
    
    
### Begin command Std_Export
__objs__ = []
__objs__.append(FreeCAD.getDocument("Unnamed").getObject("FEMMeshNetgen"))
Fem.export(__objs__, u"C:/Work/The University of Manchester Dropbox/Matt Roy/Robin_Laurence/LAZ3RUS/Data/show.inp")
