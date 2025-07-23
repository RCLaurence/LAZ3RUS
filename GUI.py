'''
-------------------------------------------------------------------------------
0.1 - Initial release
'''

__author__ = "M.J. Roy"
__version__ = "1.1"
__email__ = "matthew.roy@manchester.ac.uk"
__status__ = "Experimental"
__copyright__ = "(c) M. J. Roy, 2024-"

import sys, os
import subprocess as sp
import yaml
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk.util.numpy_support as v2n
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


from automatic_bead_to_finite_element_mesh import do_transform, in_poly, reduce_pnts, get_svd_orientation, get_trans_from_euler_angles, gen_fc_bead

class standalone_app(QtWidgets.QMainWindow):
    
    def __init__(self, parent=None):
        super(standalone_app, self).__init__(parent)
        self.main_window = interactor(self)
        self.setWindowTitle("LAZ3RUS GUI v%s" %(__version__))
        self.setCentralWidget(self.main_window)
        
        screen = QtWidgets.QApplication.primaryScreen()
        rect = screen.availableGeometry()
        self.setMinimumSize(QtCore.QSize(int(2*rect.width()/3), int(7*rect.height()/8)))

        frame = self.frameGeometry()
        center = QtWidgets.QDesktopWidget().availableGeometry().center()
        frame.moveCenter(center)
        self.move(frame.topLeft())

class main_window(QtWidgets.QWidget):
        
    def setup(self, parent):
        
        #create new layout to hold both VTK and Qt interactors
        mainlayout=QtWidgets.QHBoxLayout(parent)
        
        interactor_sub_layout = QtWidgets.QVBoxLayout()
        #create VTK widget
        self.vtkWidget = QVTKRenderWindowInteractor(parent)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(50)
        sizePolicy.setVerticalStretch(50)
        self.vtkWidget.setSizePolicy(sizePolicy)
        
        io_layout = QtWidgets.QGridLayout()
        self.load_button = QtWidgets.QPushButton('Load')
        self.settings_button = QtWidgets.QPushButton('Settings')
        self.transform_button = QtWidgets.QPushButton('Transform')
        self.transform_button.setToolTip('Show dialog with current matrix, closing dialog will apply it.')
        
        io_layout.addWidget(self.load_button,0,0,1,1)
        io_layout.addWidget(self.transform_button,0,1,1,1)
        io_layout.addWidget(self.settings_button,0,2,1,1)
        
        io_box = QtWidgets.QGroupBox("Input")
        io_box.setLayout(io_layout)
        
        bbox_display_layout = QtWidgets.QHBoxLayout()
        bbox_layout = QtWidgets.QGridLayout()
        self.bbox_x0 = QtWidgets.QDoubleSpinBox()
        self.bbox_x1 = QtWidgets.QDoubleSpinBox()
        self.bbox_y0 = QtWidgets.QDoubleSpinBox()
        self.bbox_y1 = QtWidgets.QDoubleSpinBox()
        self.bbox_z0 = QtWidgets.QDoubleSpinBox()
        self.bbox_z1 = QtWidgets.QDoubleSpinBox()
        self.zseg = QtWidgets.QDoubleSpinBox() #added later
        self.plate_mean = QtWidgets.QDoubleSpinBox() #added later
        self.fc_plate_w = QtWidgets.QDoubleSpinBox()
        self.fc_plate_h = QtWidgets.QDoubleSpinBox()
        self.fc_plate_t = QtWidgets.QDoubleSpinBox()
        bbox_entries = [self.bbox_x0, self.bbox_x1, self.bbox_y0, self.bbox_y1, self.bbox_z0, self.bbox_z1, self.zseg, self.plate_mean, self.fc_plate_w, self.fc_plate_h, self.fc_plate_t]
        bbox_labels = ['x0', 'x1', 'y0', 'y1', 'z0', 'z1', 'b', 'p', 'w', 'h', 't']
        for i in range(len(bbox_entries)):
            bbox_entries[i].setPrefix('%s = '%bbox_labels[i])
            bbox_entries[i].setSuffix(' mm')
            bbox_entries[i].setMaximum(10000)
            bbox_entries[i].setMinimum(-10000)
            bbox_entries[i].setDecimals(3)
        
        display_layout = QtWidgets.QVBoxLayout()
        self.draw_highlight_rb = QtWidgets.QCheckBox("Highlight")
        self.draw_highlight_rb.setChecked(True)
        self.draw_highlight_rb.setToolTip('Draws bounding box')
        self.op_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.op_slider.setToolTip("Change opacity of active points")
        self.op_slider.setRange(0,100)
        self.op_slider.setSliderPosition(100)
        display_layout.addWidget(self.draw_highlight_rb)
        display_layout.addWidget(self.op_slider)
        
        
        bbox_layout.addWidget(self.bbox_x0,0,0,1,1)
        bbox_layout.addWidget(self.bbox_x1,0,1,1,1)
        bbox_layout.addWidget(self.bbox_y0,1,0,1,1)
        bbox_layout.addWidget(self.bbox_y1,1,1,1,1)
        bbox_layout.addWidget(self.bbox_z0,2,0,1,1)
        bbox_layout.addWidget(self.bbox_z1,2,1,1,1)
        
        bbox_display_layout.addLayout(bbox_layout)
        bbox_display_layout.addLayout(display_layout)
        
        
        bounding_box = QtWidgets.QGroupBox("Bounding box & display")
        bounding_box.setLayout(bbox_display_layout)
        
        picking_layout = QtWidgets.QGridLayout()
        self.active_picking_indicator = QtWidgets.QLabel('Active')
        self.active_picking_indicator.setStyleSheet("background-color : gray; color : darkGray;")
        self.active_picking_indicator.setAlignment(QtCore.Qt.AlignCenter)
        self.active_picking_indicator.setToolTip('Press R with interactor in focus to activate/deactivate manual x & y point selection')
        self.active_picking_indicator.setEnabled(False)
        self.undo_last_pick_button=QtWidgets.QPushButton('Undo')
        self.undo_last_pick_button.setToolTip('Undo selection')
        self.invert_plate_sel=QtWidgets.QCheckBox("Invert selection")
        self.invert_plate_sel.setToolTip("Invert the selection")
        self.level_button = QtWidgets.QPushButton('Level')
        self.level_button.setToolTip('Level points based on selection')
        self.crop_button=QtWidgets.QPushButton('Crop')
        self.crop_button.setToolTip('Remove selected points and update')
        
        picking_layout.addWidget(self.active_picking_indicator,0,0,1,1)
        picking_layout.addWidget(self.undo_last_pick_button,0,1,1,1)
        picking_layout.addWidget(self.invert_plate_sel,1,0,1,1)
        picking_layout.addWidget(self.level_button,0,2,1,1)
        picking_layout.addWidget(self.crop_button,1,2,1,1)
        
        picking_box = QtWidgets.QGroupBox("Selection options")
        picking_box.setLayout(picking_layout)
        
        bead_fit_layout = QtWidgets.QHBoxLayout()
        self.zseg.setToolTip("z value for bead orientation")
        self.bead_orientation_button = QtWidgets.QPushButton("Orient")
        self.bead_orientation_button.setToolTip('Display orientation of weld bead')
        self.make_vertical=QtWidgets.QCheckBox("Rotate")
        self.make_vertical.setToolTip("Pressing orient will rotate the data set vertical to be in-line with y-axis")
        
        bead_fit_layout.addWidget(self.zseg)
        bead_fit_layout.addStretch()
        bead_fit_layout.addWidget(self.make_vertical)
        bead_fit_layout.addWidget(self.bead_orientation_button)
        
        plate_param_layout = QtWidgets.QHBoxLayout()
        self.get_plate_from_bb = QtWidgets.QPushButton("Select")
        self.get_plate_from_bb.setToolTip("Get plate height 'p' from bounding box")
        self.seg_width = QtWidgets.QDoubleSpinBox()
        self.seg_width.setPrefix('s = ')
        self.seg_width.setSuffix(' mm')
        self.seg_width.setMaximum(1)
        self.seg_width.setMinimum(0.01)
        self.seg_width.setDecimals(3)
        self.seg_width.setValue(0.2)
        self.get_plate_button = QtWidgets.QPushButton("Fit bead")
        
        self.plate_mean.setSingleStep(0.1)
        plate_param_layout.addWidget(self.plate_mean)
        plate_param_layout.addWidget(self.get_plate_from_bb)
        plate_param_layout.addWidget(self.seg_width)
        plate_param_layout.addWidget(self.get_plate_button)
        plate_param_layout.addStretch()
        
        bead_fit_box_layout = QtWidgets.QVBoxLayout()
        bead_fit_box_layout.addLayout(bead_fit_layout)
        bead_fit_box_layout.addLayout(plate_param_layout)
        
        bead_fit_box = QtWidgets.QGroupBox("Bead fitting")
        bead_fit_box.setLayout(bead_fit_box_layout)
        
        gen_bead_box_layout = QtWidgets.QGridLayout()
        gen_bead_box = QtWidgets.QGroupBox("FreeCAD")
        self.pbar = QtWidgets.QProgressBar(self, textVisible=True)
        self.pbar.setAlignment(Qt.AlignCenter)
        self.pbar.setFormat("Idle")
        self.pbar.setValue(0)
        self.run_button = QtWidgets.QPushButton('Run')
        
        self.invert_generated_geo = QtWidgets.QCheckBox("Invert transform")
        self.invert_generated_geo.setToolTip("Generate geometry in original orientation")
        self.load_stl_button = QtWidgets.QPushButton("Load STL")
        self.load_stl_button.setToolTip("Load the generated STL file")
        self.load_stl_button.setEnabled(False)
        
        plate_offset = [self.fc_plate_w, self.fc_plate_h, self.fc_plate_t]
        tt = ['Offset generated plate from bead start in x', 'Offset generated plate from bead start in y', 'Offset generated plate from bead start in z']
        for entry in range(len(plate_offset)):
            plate_offset[entry].setToolTip(tt[entry])
        gen_bead_box_layout.addWidget(self.fc_plate_w,0,0,1,1)
        gen_bead_box_layout.addWidget(self.fc_plate_h,0,1,1,1)
        gen_bead_box_layout.addWidget(self.fc_plate_t,0,2,1,1)
        self.fc_plate_w.setValue(30)
        self.fc_plate_h.setValue(10)
        self.fc_plate_t.setValue(1)
        gen_bead_box_layout.addWidget(self.pbar,1,0,1,2)
        gen_bead_box_layout.addWidget(self.run_button,1,2,1,1)
        gen_bead_box_layout.addWidget(self.invert_generated_geo,2,0,1,1)
        gen_bead_box_layout.addWidget(self.load_stl_button,2,2,1,1)
        
        gen_bead_box.setLayout(gen_bead_box_layout)
        
        interactor_sub_layout.addWidget(io_box)
        interactor_sub_layout.addWidget(bounding_box)
        interactor_sub_layout.addWidget(picking_box)
        interactor_sub_layout.addWidget(bead_fit_box)
        interactor_sub_layout.addWidget(gen_bead_box)
        interactor_sub_layout.addStretch()
        
        mainlayout.addWidget(self.vtkWidget)
        mainlayout.addLayout(interactor_sub_layout)
        

class interactor(QtWidgets.QWidget):
    '''
    Inherits most properties from Qwidget, but primes the VTK window, and ties functions and methods to interactors defined in main_window
    '''
    def __init__(self,parent):
        super(interactor, self).__init__(parent)
        self.ui = main_window()
        self.ui.setup(self)
        self.ren = vtk.vtkRenderer()
        colors = vtk.vtkNamedColors()
        self.ren.SetBackground(colors.GetColor3d("aliceblue"))

        self.ui.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.cam_orient_manipulator = vtk.vtkCameraOrientationWidget()
        self.cam_orient_manipulator.SetParentRenderer(self.ren)

        self.iren = self.ui.vtkWidget.GetRenderWindow().GetInteractor()
        style=vtk.vtkInteractorStyleRubberBandPick()
        self.iren.SetInteractorStyle(style)
        self.iren.AddObserver("KeyPressEvent", self.keypress)

        picker = vtk.vtkRenderedAreaPicker()
        self.iren.SetPicker(picker)
        picker.AddObserver(vtk.vtkCommand.EndPickEvent, self.picker_callback)

        self.ren.GetActiveCamera().ParallelProjectionOn()
        self.ren.Render()
        self.ui.vtkWidget.Initialize()
        

        self.settings_file_name = None
        self.data_file_name = None
        self.transform = np.eye(4)
        self.picking = False
        
        self.ui.load_button.clicked.connect(self.load_file)
        self.ui.settings_button.clicked.connect(self.check_settings)
        self.ui.transform_button.clicked.connect(self.get_transform)
        self.ui.undo_last_pick_button.clicked.connect(self.undo_crop)
        self.ui.crop_button.clicked.connect(self.crop_bb)
        self.ui.level_button.clicked.connect(self.apply_level)
        self.ui.bead_orientation_button.clicked.connect(self.orient)
        self.ui.get_plate_from_bb.clicked.connect(self.plate_select)
        self.ui.get_plate_button.clicked.connect(self.plate)
        self.ui.op_slider.valueChanged[int].connect(self.change_opacity)
        self.ui.run_button.clicked.connect(self.run_fc)
        self.ui.load_stl_button.clicked.connect(self.load_stl)

    def keypress(self, obj, event):
        key = obj.GetKeyCode()
        if key == 's':
            sw = settings_widget(self)
            sw.exec_()
            self.check_settings()
        elif key == "1":
            xyview(self.ren)
        elif key == "2":
            yzview(self.ren)
        elif key == "3":
            xzview(self.ren)
        
        elif key == 'r':
            if self.picking:
                self.picking = False
                self.ui.active_picking_indicator.setStyleSheet("background-color : gray; color : darkGray;")
            else:
                self.picking = True
                self.ui.active_picking_indicator.setStyleSheet("background-color :rgb(77, 209, 97);")
            
        self.ui.vtkWidget.update()        

    def load_file(self):
        '''
        Loads an xyz file
        '''
        if self.settings_file_name is not None:
            fname = get_file('*.xyz', read_config(self.settings_file_name)[0])
        else:
            fname = get_file('*.xyz')
        
        if fname is None:
            return
            
        self.points = np.genfromtxt(fname, delimiter=' ', skip_header=0, usecols=(0, 1, 2))
        self.undo_crop()
        
    
    def check_settings(self):
        if self.settings_file_name is None:
            self.work_dir, self.freecad_cmd, fname = read_config(None)
            if fname is None or not(os.path.isfile(fname)):
                return
            else: self.settings_file_name = fname
        else:
            self.work_dir, self.freecad_cmd, _ = read_config(self.settings_file_name)
    
    def get_transform(self):
        if self.settings_file_name is not None:
            mw = matrix_widget(self, read_config(self.settings_file_name)[0], self.transform)
        else:
            mw = matrix_widget(self, os.getcwd(), self.transform)
        mw.exec_()
        
        if not hasattr(self, 'points'):
            return
        
        self.points = do_transform(self.points, mw.matrix @ np.linalg.inv(self.transform))
        self.transform = mw.matrix
        self.draw_points()
        
    def draw_points(self):
        
        self.ren.RemoveAllViewProps()
        
        self.point_actor, \
        self.pnt_polydata, \
        self.colors, lut = \
        gen_point_cloud(self.points[self.active_pnt],None,None)

        sb_widget = gen_scalar_bar()
        sb_widget.SetInteractor(self.iren)
        sb_widget.On()
        self.sb_actor = sb_widget.GetScalarBarActor()
        self.sb_actor.SetLookupTable(lut)
        self.ren.AddActor(self.sb_actor)

        self.ren.AddActor(self.point_actor)

        limits = get_limits(self.points[self.active_pnt], 0)
        self.ui.bbox_x0.setValue(limits[0])
        self.ui.bbox_x1.setValue(limits[1])
        self.ui.bbox_y0.setValue(limits[2])
        self.ui.bbox_y1.setValue(limits[3])
        self.ui.bbox_z0.setValue(limits[4])
        self.ui.bbox_z1.setValue(limits[5])
        self.ui.zseg.setValue((limits[4] + limits[5]) / 2)
        
        self.change_opacity(self.ui.op_slider.value())

        self.ren.ResetCamera()
        self.ui.vtkWidget.update()

    def change_opacity(self,value):
        if hasattr(self,'point_actor'):
            self.point_actor.GetProperty().SetOpacity(value/100)
        self.ui.vtkWidget.update()

    def picker_callback(self,obj,event):
        '''
        Manual picking callback function
        '''
        extract = vtk.vtkExtractSelectedFrustum()
        f_planes=obj.GetFrustum() #collection of planes based on unscaled display
        planes=vtk.vtkPlanes()
        normals=vtk.vtkDoubleArray()
        normals.SetNumberOfComponents(3)
        normals.SetNumberOfTuples(6)
        origins=vtk.vtkPoints()
        for j in range(6):
            i=f_planes.GetPlane(j)
            k=i.GetOrigin()
            q=i.GetNormal()
            origins.InsertNextPoint(k[0],k[1],k[2])
            normals.SetTuple(j,(q[0],q[1],q[2]))
        planes.SetNormals(normals)
        planes.SetPoints(origins)

        extract.SetFrustum(planes)
        extract.SetInputData(self.pnt_polydata)
        extract.Update()
        extracted = extract.GetOutput()

        ids = vtk.vtkIdTypeArray()
        ids = extracted.GetPointData().GetArray("vtkOriginalPointIds")

        if ids:
            selected_points = []
            for i in range(ids.GetNumberOfTuples()):
                selected_points.append(self.points[self.active_pnt[ids.GetValue(i)],:])
            selected_points = np.asarray(selected_points)
            limits = get_limits(selected_points, 0)
            self.ui.bbox_x0.setValue(limits[0])
            self.ui.bbox_x1.setValue(limits[1])
            self.ui.bbox_y0.setValue(limits[2])
            self.ui.bbox_y1.setValue(limits[3])
            self.ui.bbox_z0.setValue(limits[4])
            self.ui.bbox_z1.setValue(limits[5])
            self.ui.zseg.setValue((limits[4] + limits[5]) / 2)
            self.draw_box()
        
    def draw_box(self):
        if hasattr(self, 'id_box_actor'):
            self.ren.RemoveActor(self.id_box_actor)
            
        coords = np.array([[self.ui.bbox_x0.value(),self.ui.bbox_y0.value(),self.ui.bbox_z0.value()],
        [self.ui.bbox_x0.value(),self.ui.bbox_y1.value(),self.ui.bbox_z0.value()],
        [self.ui.bbox_x1.value(),self.ui.bbox_y1.value(),self.ui.bbox_z0.value()],
        [self.ui.bbox_x1.value(),self.ui.bbox_y0.value(),self.ui.bbox_z0.value()],
        [self.ui.bbox_x0.value(),self.ui.bbox_y0.value(),self.ui.bbox_z1.value()],
        [self.ui.bbox_x0.value(),self.ui.bbox_y1.value(),self.ui.bbox_z1.value()],
        [self.ui.bbox_x1.value(),self.ui.bbox_y1.value(),self.ui.bbox_z1.value()],
        [self.ui.bbox_x1.value(),self.ui.bbox_y0.value(),self.ui.bbox_z1.value()]])
        
        idx = np.array([[0, 1, 2, 3, 0], [3, 2, 6, 7, 3], [0, 4, 7, 3, 0], [0, 1, 5, 4, 0], [4, 5, 6, 7, 4], [1, 5, 6, 2, 1]])
        
        self.box = coords[idx[0],:]
        
        if self.ui.draw_highlight_rb.isChecked():
            self.id_box_actor = vtk.vtkAssembly()
            for face in idx:
                self.id_box_actor.AddPart(gen_outline(coords[face,:],(0,0,0))[0])
            self.ren.AddActor(self.id_box_actor)
    
    def crop_bb(self):
        self.active_pnt = self.active_pnt[
        in_poly(self.box[:,0:2],self.points[self.active_pnt,:]) & 
        (self.points[self.active_pnt,2]<self.ui.bbox_z1.value()) & 
        (self.points[self.active_pnt,2]>self.ui.bbox_z0.value())
        ]
        self.ui.zseg.setValue((self.ui.bbox_z1.value() + self.ui.bbox_z0.value()) / 2)
        self.draw_points()

    def apply_level(self):
        
        if not hasattr(self, 'box'):
            return
        
        #get the points from the current bounding box
        
        points_avail = self.points[self.active_pnt, :]
        if not self.ui.invert_plate_sel.isChecked():
            target_points = points_avail[in_poly(self.box[:,0:2], points_avail),:]
        else:
            target_points = points_avail[~in_poly(self.box[:,0:2], points_avail),:]
        R = get_svd_orientation(target_points[reduce_pnts(target_points),:])
        trans = np.eye(4)
        trans[0:3, 0:3] = R
        self.points = do_transform(self.points, trans)
        self.transform = trans @ self.transform
        self.ren.RemoveActor(self.id_box_actor)
        self.draw_points()
    
    def orient(self):
        
        if hasattr(self, 'orient_actor'):
            self.ren.RemoveActor(self.orient_actor)
        
        bp = self.points[self.active_pnt][self.points[self.active_pnt][:,2] > self.ui.zseg.value()]
        
        coeff = np.polyfit(bp[:, 0], bp[:, 1], 1)
        
        x = np.linspace(np.min(bp[:, 0]), np.max(bp[:, 0]),200).reshape(-1,1)
        line = np.hstack((x, x * coeff[0] + coeff[1], self.ui.zseg.value()*np.ones(len(x)).reshape(-1,1)))
        self.orient_actor = gen_outline(line,(0,0,0),10)[0]
        self.ren.AddActor(self.orient_actor)
        
        self.ui.vtkWidget.update()
        
        if self.ui.make_vertical.isChecked():
            self.ren.RemoveActor(self.orient_actor)
            alpha = -np.arctan(coeff[0]) + np.pi / 2  # alpha returns to the x axis, with an additional cw 90°
            rotate_xy = get_trans_from_euler_angles(0, 0, alpha)
            self.points = do_transform(self.points, rotate_xy)
            self.transform = self.transform @ rotate_xy
            self.draw_points()

    def plate_select(self):
        if not hasattr(self, 'box'):
            return
        
        points_avail = self.points[self.active_pnt, :]
        if not self.ui.invert_plate_sel.isChecked():
            target_points = points_avail[in_poly(self.box[:,0:2], points_avail),:]
        else:
            target_points = points_avail[~in_poly(self.box[:,0:2], points_avail),:]
        self.ui.plate_mean.setValue(np.mean(target_points[:,2]))

    def plate(self):
        
        def func(x, a, h, k, level_z):
            '''
            Description of a piecewise/capped parabola of the form y = max(a*(x-h)**2 + k, level_z)
            '''
            y = np.zeros_like(x)
            for i in range(len(x)):
                y[i] = np.max(np.append(a * (x[i] - h) ** 2 + k, level_z))
            return y
        
        if hasattr(self, 'fit_actor_list') and self.fit_actor_list:
            for actor in self.fit_actor_list:
                self.ren.RemoveActor(actor)
        self.fit_actor_list = []
        
        height = self.ui.plate_mean.value()
        points_avail = self.points[self.active_pnt, :]
        
        cent = np.mean(points_avail, axis = 0)
        seg_t = self.ui.seg_width.value()
        mask = np.logical_and(points_avail[:, 1] < (cent[1] + seg_t), points_avail[:, 1] > (cent[1] - seg_t))
        x_sec = points_avail[mask, :]
        sl_x = np.linspace(np.min(x_sec[:, 0]), np.max(x_sec[:, 0]))
        p, _ = find_peaks(x_sec[:, 2], prominence=1)  # to provide guess initialize h,k for fitting parabola
        
        # perform curve fit with locked height with an initial guess of a=-0.5, h & k equal to the peaks
        popt, _ = curve_fit(lambda x, a, h, k: func(x, a, h, k, height), x_sec[:, 0], x_sec[:, 2],
                            p0=np.array([-0.5, x_sec[p[0], 0], x_sec[p[0], 2]]))

        fit = func(sl_x, *popt, height)  # result of the fit

        # get width in x of bead, will run between inter_x[0] to inter_x[1]
        p_ = [popt[0], -2 * popt[0] * popt[1],
              popt[0] * popt[1] ** 2 + popt[2]]  # standard representation of vertex representation above
        inter_x = np.roots(np.asarray(p_) - np.asarray([0, 0, height]))

        #plot with an outline
        line = np.column_stack((sl_x,np.ones(len(sl_x))*cent[1],fit))
        
        #find the extents of the bead using the maximum height of the plate
        bp = points_avail[(points_avail[:, 2] > (height+seg_t)), :]

        y_s, y_e = np.max(bp[:, 1]), np.min(bp[:, 1])
        x = np.linspace(inter_x[0], inter_x[1])
        # get x,y extents of bead
        start_xy = np.column_stack((x, (popt[0] * (x - popt[1]) ** 2 + y_s), np.ones(len(x)) * height))
        end_xy = np.column_stack((np.flip(x), (-popt[0] * (x - popt[1]) ** 2 + y_e), np.ones(len(x)) * height))
        
        #translate to start of the bead
        target = np.array([popt[1], start_xy[0, 1], popt[2] + (1 / (4 * popt[0]))])
        translate = np.eye(4)
        translate[0:3, -1] = -target
        # update h,k values in popt, y pad end value, inter_x values
        inter_x = inter_x - popt[1]
        # height = height - (popt[2] + (1 / (4 * popt[0])))
        y_e = end_xy[0, 1] - start_xy[0, 1]
        popt[1], popt[2] = 0, -1 / (4 * popt[0])
        
        self.points = do_transform(self.points, translate)
        start_xy = do_transform(start_xy,translate)
        end_xy = do_transform(end_xy,translate)
        line = do_transform(line,translate)
        self.transform = self.transform @ translate
        self.ui.plate_mean.setValue(0)
        self.draw_points()
        
        xy_outline = np.vstack((start_xy, end_xy, start_xy[0, :]))
        
        fit_actor = gen_outline(line,(0,0,0),10)[0]
        outline_fit_actor = gen_outline(xy_outline,(0.5,0.5,0.5),10)[0]
        weld_path_actor = gen_line_actor([0,0,0], [0,y_e,0])
        
        self.fit_actor_list = [fit_actor, outline_fit_actor, weld_path_actor]
        for actor in self.fit_actor_list:
            self.ren.AddActor(actor)
        self.ren.RemoveActor(self.id_box_actor)
        
        self.ui.vtkWidget.update()
        
        #return inter_x, popt and y_e to main object
        self.inter_x = inter_x
        self.popt = popt
        self.y_e = y_e
        
        
    def undo_crop(self):
        '''
        Undoes the last pick registered
        '''
        if hasattr(self, 'id_box_actor'):
            self.ren.RemoveActor(self.id_box_actor)
        self.active_pnt = np.arange(0, len(self.points), 1, dtype=int)
        self.draw_points()

    def run_fc(self):

        # inter_x and popt will exist if y_e does
        if not hasattr(self,'y_e'):
            return
        
        if not hasattr(self,'freecad_cmd'):
            self.check_settings()
        
        if not hasattr(self,'macro_fname'):
            self.macro_fname = get_save_file('*.py',self.work_dir)
        
        height = self.ui.fc_plate_h.value()
        width = self.ui.fc_plate_w.value()
        thickness = self.ui.fc_plate_t.value()
        
        if self.ui.invert_generated_geo.isChecked():
            self.inverted_geo = True
            gen_fc_bead(self.inter_x, self.y_e, self.popt, [width, height, thickness], self.transform, self.macro_fname)
        else:
            self.inverted_geo = False
            gen_fc_bead(self.inter_x, self.y_e, self.popt, [width, height, thickness], np.eye(4), self.macro_fname)

        self.thread = execute_fc(self.macro_fname, self.freecad_cmd)

        self.thread._signal.connect(self.signal_accept)
        self.thread.start()
        self.ui.pbar.setTextVisible(True)
        self.ui.pbar.setStyleSheet("")
        self.ui.pbar.setRange(0,0)
        
    def signal_accept(self, msg):
        if int(msg) == 100:
            self.ui.pbar.setRange(0,100)
            self.ui.pbar.setValue(0)
            self.ui.pbar.setFormat("Complete")
            self.ui.pbar.setStyleSheet("QProgressBar"
              "{"
              "background-color: lightgreen;"
              "border : 1px"
              "}")
            self.ui.load_stl_button.setEnabled(True)
            
    def load_stl(self):
        
        if hasattr(self,'stl_actor'):
            self.ren.RemoveActor(self.stl_actor)
        
        if self.inverted_geo:
            self.points = do_transform(self.points, np.linalg.inv(self.transform))
            self.transform = np.eye(4)
            self.draw_points()
        
        filep = self.macro_fname[:-2] + 'stl'
        print(filep)
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filep)
        reader.Update()
        
        stl_polydata = reader.GetOutput()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(stl_polydata)
        self.stl_actor = vtk.vtkActor()
        self.stl_actor.SetMapper(mapper)
        self.stl_actor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d('Gray'))
        self.stl_actor.GetProperty().SetOpacity(50)
        self.ren.AddActor(self.stl_actor)
        self.ui.vtkWidget.update()

def get_save_file(*args):
    '''
    Returns absolute path to filename and the directory it is located in from a PyQt5 filedialog. First value is file extension, second is a string which overwrites the window message.
    '''
    ext = args[0]
    if len(args)>1:
        dir = args[1]
    else: dir = os.getcwd()
    
    file_type_names = {}
    file_type_names['*.py'] = 'FreeCAD Python macro'

    filer = QtWidgets.QFileDialog.getSaveFileName(None, "Save as:", \
    dir, \
    str(file_type_names[ext]+' ('+ext+')') \
    )
    
    if filer[0] == '':
        return None, None
    else:
        return filer[0]

def get_file(*args):
    '''
    Returns absolute path to filename and the directory it is located in from a PyQt5 filedialog. First value is file extension, second is a string which overwrites the window message.
    '''
    ext = args[0]
    if len(args)>1:
        launchdir = args[1]
    else: launchdir = os.getcwd()
    ftypeName={}
    ftypeName['*.txt']=["Select text file:", "*.txt", "TXT File"]
    ftypeName['*.xyz']=["Select pointcloud:", "*.xyz", "XYZ File"]
    ftypeName['*.yaml']=["Select settings file:", "*.yaml", "YAML File"]
    ftypeName['*.*']=["Select external executable:", "*.*", "..."]
    
    filer = QtWidgets.QFileDialog.getOpenFileName(None, ftypeName[ext][0], 
         launchdir,(ftypeName[ext][2]+' ('+ftypeName[ext][1]+');;All Files (*.*)'))
    
    if filer[0] == '':
        return None
    else:
        return filer[0]

def get_directory():

    directory = str(QtWidgets.QFileDialog.getExistingDirectory(None, os.getcwd(), "Select Directory"))
    
    if directory == '':
        return None
    else:
        return directory

def read_config(file):
    
    if file is None:
        fname = get_file("*.yaml")
        
        if fname is None or not(os.path.isfile(fname)):
            return
    else:
        fname = file
    
    with open(fname, 'r') as f:
        read = yaml.safe_load(f)
    
    return read['filenames']['dir'], read['freecad']['exec'], fname

            
class matrix_widget(QtWidgets.QDialog):
    def __init__(self, parent, start_dir, matrix):
        super(matrix_widget, self).__init__(parent)
        
        self.wd = start_dir
        self.matrix = matrix
        
        self.setWindowTitle("LAZ3RUS current transformation matrix")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(QtCore.QSize(450, 200))
        
        mainlayout = QtWidgets.QVBoxLayout()
        
        self.table = QtWidgets.QTableWidget()
        self.table.setToolTip("Current transformation matrix")
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        
        
        self.table.setRowCount(self.matrix.shape[1])
        self.table.setColumnCount(self.matrix.shape[0])
        self.populate_matrix()
        
        button_layout = QtWidgets.QHBoxLayout()
        load_button = QtWidgets.QPushButton('Load')
        load_button.setToolTip("Loading an additional file containing a matrix will be applied to the current.")
        reset_button = QtWidgets.QPushButton('Reset')
        
        mainlayout.addWidget(self.table)
        button_layout.addWidget(load_button)
        button_layout.addWidget(reset_button)
        mainlayout.addLayout(button_layout)
        self.setLayout(mainlayout)
        
        load_button.clicked.connect(self.load_matrix)
        reset_button.clicked.connect(self.reset_matrix)

        self.show()
        
    def populate_matrix(self):
        
        for i,row in enumerate(self.matrix):
            for j,val in enumerate(row):
                self.table.setItem(i,j,QtWidgets.QTableWidgetItem(str(val)))
        
    def load_matrix(self):
        '''
        Loads a matrix from a txt file
        '''
        fname = get_file('*.txt', self.wd)
        
        if fname is None:
            return
            
        else:
            self.matrix = np.loadtxt(fname) @ self.matrix
            self.populate_matrix()

    def reset_matrix(self):
        self.matrix = np.eye(4)
        self.populate_matrix()
    

class settings_widget(QtWidgets.QDialog):

    def __init__(self, parent):
        super(settings_widget, self).__init__(parent)
        
        self.setWindowTitle("LAZ3RUS file IO settings")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(QtCore.QSize(450, 200))
        
        param_layout = QtWidgets.QGridLayout()
        
        work_dir_path_label = QtWidgets.QLabel('Working directory:')
        self.work_dir_path = QtWidgets.QLineEdit()
        wd_choose_path = QtWidgets.QPushButton('...')
        wd_choose_path.setMaximumWidth(30)
        wd_choose_path.setAutoDefault(False)
        
        fc_exec_path_label = QtWidgets.QLabel('FreeCAD executable:')
        self.fc_exec_path = QtWidgets.QLineEdit()
        fc_choose_path = QtWidgets.QPushButton('...')
        fc_choose_path.setMaximumWidth(30)
        fc_choose_path.setAutoDefault(False)
        
        param_layout.addWidget(work_dir_path_label,0,0,1,1)
        param_layout.addWidget(self.work_dir_path,0,1,1,2)
        param_layout.addWidget(wd_choose_path,0,3,1,1)
        
        param_layout.addWidget(fc_exec_path_label,1,0,1,1)
        param_layout.addWidget(self.fc_exec_path,1,1,1,2)
        param_layout.addWidget(fc_choose_path,1,3,1,1)
        
        self.setLayout(param_layout)
        
        self.settings_file_name = None
        fc_choose_path.clicked.connect(self.set_freecad)
        wd_choose_path.clicked.connect(self.set_wd)
        
        self.set_config()
        self.show()

    def set_config(self):
        
        if self.settings_file_name is None:
            wd, fc, fname = read_config(None)
            if fname is None or not(os.path.isfile(fname)):
                return
            else: self.settings_file_name = fname
        else:
            wd, fc, _ = read_config(self.settings_file_name)
        
        self.work_dir_path.setText(wd)
        self.fc_exec_path.setText(fc)

    def set_wd(self):
        wd = get_directory()
        if wd is None:
            return
        self.work_dir_path.setText(wd)
        self.make_config_change()

    def set_freecad(self):
        f = get_file('*.*')
        if f is None or not(os.path.isfile(f)):
            return
        self.fc_exec_path.setText(f)

    def make_config_change(self):
        
        with open(self.settings_file_name, 'r') as yamlfile:
            cur_yaml = yaml.safe_load(yamlfile)
            cur_yaml['filenames']['dir'] = str(self.work_dir_path.text())
            cur_yaml['freecad']['exec'] = str(self.fc_exec_path.text())
        if cur_yaml:
            with open(self.settings_file_name, 'w') as yamlfile:
                yaml.safe_dump(cur_yaml, yamlfile)

    def closeEvent(self, *args, **kwargs):
        super(QtWidgets.QDialog, self).closeEvent(*args, **kwargs)
        self.make_config_change()

def gen_point_cloud(pts,color=None,r=None,size=2):
    '''
    Returns vtk objects and actor for a point cloud having size points, returns color array associated with the actor/polydata object as well as a lookuptable for rendering a scalebar if colouration is applied based on height. color needs to be specified as an RGB 0-255 tuple.
    '''
    
    lut=None
    vtkPnts = vtk.vtkPoints()
    vtkVerts = vtk.vtkCellArray()
    
    #load up points
    vtkPnts.SetData(v2n.numpy_to_vtk(pts))

    for i in np.arange(len(pts)):
        vtkVerts.InsertNextCell(1)
        vtkVerts.InsertCellPoint(i)

    pC = vtk.vtkPolyData()
    pC.SetPoints(vtkPnts)
    pC.SetVerts(vtkVerts)

    mapper = vtk.vtkDataSetMapper()

    if color is None:
        vtk_z_array = v2n.numpy_to_vtk(pts[:,-1])
        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.667, 0.0)
        if r is None:
            lut.SetTableRange(np.amin(pts[:,-1]), np.amax(pts[:,-1]))
        else:
            lut.SetTableRange(r[0],r[1])
        lut.Build()
        colors = lut.MapScalars(vtk_z_array,vtk.VTK_COLOR_MODE_DEFAULT,-1,vtk.VTK_RGB)

        
    elif isinstance(color,tuple):
        #if assigning single color every point, lut will return as None
        colors=vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        for i in np.arange(len(pts)):
            colors.InsertNextTuple(color)
        pC.GetPointData().SetScalars(colors)
        mapper.SetInputData(pC)
        
    elif isinstance(color, str):
        vtk_z_array = v2n.numpy_to_vtk(pts[:,-1])
        lut = get_diverging_lut(color) #add options here for other baseline color series
        if r is None:
            lut.SetTableRange(np.amin(pts[:,-1]), np.amax(pts[:,-1]))
        else:
            lut.SetTableRange(r[0],r[1])
        lut.Build()
        colors = lut.MapScalars(vtk_z_array,vtk.VTK_COLOR_MODE_DEFAULT,-1,vtk.VTK_RGB)

    
    pC.GetPointData().SetScalars(colors)
    mapper.SetInputData(pC)

    actor=vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetPointSize(size)
    return actor, pC, colors, lut

def xyview(ren):
    camera = ren.GetActiveCamera()
    camera.SetPosition(0,0,1)
    camera.SetFocalPoint(0,0,0)
    camera.SetViewUp(0,1,0)
    ren.ResetCamera()

def yzview(ren):
    camera = ren.GetActiveCamera()
    camera.SetPosition(1,0,0)
    camera.SetFocalPoint(0,0,0)
    camera.SetViewUp(0,0,1)
    ren.ResetCamera()

def xzview(ren):
    vtk.vtkObject.GlobalWarningDisplayOff() #mapping from '3' triggers an underlying stereoview that most displays do not support for trackball interactors
    camera = ren.GetActiveCamera()
    camera.SetPosition(0,1,0)
    camera.SetFocalPoint(0,0,0)
    camera.SetViewUp(0,0,1)
    ren.ResetCamera()

def gen_scalar_bar(title = None, num_contours = 13, side = 'left'):
    '''
    Returns a formatted scalebar widget based on the incoming lookup table, title and number of labels to the left or right of the interacator depending on 'side'
    '''
    bar_widget = vtk.vtkScalarBarWidget()
    scalarBarRep = bar_widget.GetRepresentation()
    if side == 'left':
        scalarBarRep.GetPositionCoordinate().SetValue(0.005,0.01) #bottom left
        scalarBarRep.GetPosition2Coordinate().SetValue(0.095,0.98) #top right
    elif side == 'right':
        scalarBarRep.GetPositionCoordinate().SetValue(0.903,0.01) #bottom left
        scalarBarRep.GetPosition2Coordinate().SetValue(0.095,0.98) #top right
    sb_actor=bar_widget.GetScalarBarActor()

    sb_actor.SetTitle(title)
    sb_actor.SetNumberOfLabels(num_contours)

    #attempt to change scalebar properties
    sb_actor.GetLabelTextProperty().SetColor(0,0,0)
    sb_actor.GetTitleTextProperty().SetColor(0,0,0)
    sb_actor.GetLabelTextProperty().SetFontSize(1)
    sb_actor.GetTitleTextProperty().SetFontSize(1)
    sb_actor.SetLabelFormat("%.3f")

    return bar_widget

def get_limits(pts, factor = 0.1):
    '''
    Returns a bounding box with x,y values bumped out by factor
    '''
    RefMin = np.amin(pts,axis=0)
    RefMax = np.amax(pts,axis=0)

    extents=RefMax-RefMin #extents
    rl=factor*(np.amin(extents[0:2])) #linear 'scale' to set up interactor
    return [RefMin[0]-rl, \
      RefMax[0]+rl, \
      RefMin[1]-rl, \
      RefMax[1]+rl, \
      RefMin[2],RefMax[2]]

def gen_outline(pts, color = (1,1,1), size = 2):
    '''
    Returns an outline actor with specified pts, color and size. Incoming pnts should be ordered.
    '''
    if color[0]<=1 and color != None:
        color=(int(color[0]*255),int(color[1]*255),int(color[2]*255))
    if color[0]>=1 and color != None:
        color=(color[0]/float(255),color[1]/float(255),color[2]/float(255))
    points=vtk.vtkPoints()

    points.SetData(v2n.numpy_to_vtk(pts))

    lineseg=vtk.vtkPolygon()
    lineseg.GetPointIds().SetNumberOfIds(len(pts))
    for i in range(len(pts)):
        lineseg.GetPointIds().SetId(i,i)
    linesegcells=vtk.vtkCellArray()
    linesegcells.InsertNextCell(lineseg)
    outline=vtk.vtkPolyData()
    outline.SetPoints(points)
    outline.SetVerts(linesegcells)
    outline.SetLines(linesegcells)
    Omapper=vtk.vtkPolyDataMapper()
    Omapper.SetInputData(outline)
    outlineActor=vtk.vtkActor()
    outlineActor.SetMapper(Omapper)
    outlineActor.GetProperty().SetColor(color)
    outlineActor.GetProperty().SetPointSize(size)
    outlineActor.GetProperty().SetLineWidth(size)
    return outlineActor, outline

def gen_line_actor(p1, p2, res = 50, color = None, radius = 0.5):
    '''
    Returns a line running from p1 to p2, glyphed for visibility
    '''

    line_source = vtk.vtkLineSource()
    line_source.SetResolution(res)
    line_source.SetPoint1(p1)
    line_source.SetPoint2(p2)
    line_source.Update()
    
    tube_filter = vtk.vtkTubeFilter()
    tube_filter.SetInputConnection(line_source.GetOutputPort())
    tube_filter.SetRadius(radius)
    tube_filter.SetNumberOfSides(res)
    tube_filter.Update()
    
    ph_res = res
    th_res = res
    
    sphere1 = vtk.vtkSphereSource()
    sphere1.SetCenter(p1)
    sphere1.SetPhiResolution(ph_res)
    sphere1.SetThetaResolution(th_res)
    sphere1.SetRadius(radius * 2)
    sphere1.Update()
    
    sphere2 = vtk.vtkSphereSource()
    sphere2.SetCenter(p2)
    sphere2.SetPhiResolution(ph_res)
    sphere2.SetThetaResolution(th_res)
    sphere2.SetRadius(radius * 0.5)
    sphere2.Update()
    
    appendFilter = vtk.vtkAppendPolyData()
    appendFilter.AddInputData(sphere1.GetOutput())
    appendFilter.AddInputData(tube_filter.GetOutput())
    appendFilter.AddInputData(sphere2.GetOutput())
    appendFilter.Update()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(appendFilter.GetOutput())
    line_actor = vtk.vtkActor()
    line_actor.SetMapper(mapper)

    if color == None:
        line_actor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d("Violet"))
    else:
        line_actor.GetProperty().SetColor(color)
    return line_actor

class execute_fc(QThread):
    '''
    Sets up and runs external thread, emits 100 when done.
    '''
    _signal = pyqtSignal(int)
    def __init__(self,script, exe):
        super(execute_fc, self).__init__()
        self.script = script
        self.exe = exe #executable path

    def run(self):
        current_dir = os.getcwd()
        output_dir = os.path.dirname(self.script)
        base = os.path.basename(self.script)
        os.chdir(output_dir)

        try:
            print('LAZ3RUS exec: %s %s'%(self.exe,base))
            out = sp.check_output([self.exe, base], shell=True)
            print("FreeCAD output log:")
            print("----------------")
            print(out.decode("utf-8"))
            print("----------------")
            print("LAZ3RUS: FreeCAD run completed . . . Idle")
        except sp.CalledProcessError as e:
            print("FreeCAD script failed for some reason.")
            print(e)
        
        os.chdir(current_dir)
        self._signal.emit(100)

if __name__ == "__main__":
    app=QtWidgets.QApplication(sys.argv)
    window = standalone_app()
    window.show()
    window.main_window.cam_orient_manipulator.On()
    sys.exit(app.exec_())