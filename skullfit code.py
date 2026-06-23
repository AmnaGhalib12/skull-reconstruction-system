import os
import sys
import vtk
import pyvista as pv
import hashlib
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, morphology, measure
from PIL import Image
import pandas as pd
import csv
from scipy.ndimage import binary_fill_holes

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QFileDialog, QLabel, QGraphicsPixmapItem, QHBoxLayout, QMessageBox,
                             QProgressBar, QFrame, QGroupBox, QDoubleSpinBox, QSplitter, QScrollArea, QDialog, QSlider)
from PyQt5.QtCore import Qt, QSize, QTimer
from skimage.util import img_as_ubyte
from pyvistaqt import QtInteractor
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

#from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from PyQt5.QtGui import QPixmap, QFont, QIcon, QColor, QPalette, QLinearGradient, QBrush, QPainter


# Define application-wide style constants

class AppStyle:
    # Color palette
    PRIMARY_COLOR = "#1a6e9c"  # Darker blue
    SECONDARY_COLOR = "#78C5D6"  # Lighter blue
    ACCENT_COLOR = "#34c759"  # Medical green
    DANGER_COLOR = "#ff3b30"  # Medical red
    TEXT_COLOR = "#2c3e50"  # Dark blue-gray
    LIGHT_TEXT = "#ffffff"  # White text
    BACKGROUND_COLOR = "#f5f7fa"  # Light gray-blue background
    CARD_COLOR = "#ffffff"  # White card background
    BORDER_COLOR = "#e1e5eb"  # Light border

    # Typography
    TITLE_FONT = "font-family: 'Segoe UI', Arial, sans-serif; font-weight: 600;"
    BODY_FONT = "font-family: 'Segoe UI', Arial, sans-serif;"

    # Dimensions
    BUTTON_HEIGHT = "36px"
    LARGE_BUTTON_HEIGHT = "46px"
    BORDER_RADIUS = "6px"

    # Common styles
    CARD_STYLE = f"""
        background-color: {CARD_COLOR}; 
        border-radius: {BORDER_RADIUS};
        border: 1px solid {BORDER_COLOR};
        padding: 20px;
    """

    PRIMARY_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {PRIMARY_COLOR};
            color: {LIGHT_TEXT};
            border: none;
            border-radius: {BORDER_RADIUS};
            padding: 10px 20px;
            font-weight: 500;
            height: {BUTTON_HEIGHT};
            min-width: 100px;
        }}
        QPushButton:hover {{
            background-color: #2180b4;
        }}
        QPushButton:pressed {{
            background-color: #155d84;
        }}
        QPushButton:disabled {{
            background-color: #9cbfd3;
            color: #e6e6e6;
        }}
    """

    SECONDARY_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {SECONDARY_COLOR};
            color: {TEXT_COLOR};
            border: none;
            border-radius: {BORDER_RADIUS};
            padding: 10px 20px;
            font-weight: 500;
            height: {BUTTON_HEIGHT};
            min-width: 100px;
        }}
        QPushButton:hover {{
            background-color: #8ad1e0;
        }}
        QPushButton:pressed {{
            background-color: #67b6c7;
        }}
        QPushButton:disabled {{
            background-color: #c4e3ec;
            color: #8f9295;
        }}
    """

    ACTION_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: {ACCENT_COLOR};
            color: {LIGHT_TEXT};
            border: none;
            border-radius: {BORDER_RADIUS};
            padding: 10px 20px;
            font-weight: 500;
            height: {BUTTON_HEIGHT};
            min-width: 100px;
        }}
        QPushButton:hover {{
            background-color: #40d366;
        }}
        QPushButton:pressed {{
            background-color: #2aaa4a;
        }}
        QPushButton:disabled {{
            background-color: #a7e9b8;
            color: #e6e6e6;
        }}
    """

    BACK_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: #f2f2f2;
            color: {TEXT_COLOR};
            border: 1px solid #d9d9d9;
            border-radius: {BORDER_RADIUS};
            padding: 8px 15px;
            font-weight: 500;
            height: {BUTTON_HEIGHT};
        }}
        QPushButton:hover {{
            background-color: #e6e6e6;
        }}
        QPushButton:pressed {{
            background-color: #d9d9d9;
        }}
    """

    MAIN_WINDOW_STYLE = f"""
        QMainWindow, QWidget {{
            background-color: {BACKGROUND_COLOR};
            color: {TEXT_COLOR};
            {BODY_FONT}
        }}
        QLabel {{
            color: {TEXT_COLOR};
        }}
        QProgressBar {{
            border: 1px solid {BORDER_COLOR};
            border-radius: {BORDER_RADIUS};
            background-color: white;
            text-align: center;
            height: 20px;
        }}
        QProgressBar::chunk {{
            background-color: {SECONDARY_COLOR};
            border-radius: {BORDER_RADIUS};
        }}
    """


### Main Window: Skull Reconstruction ###
class SkullReconstruction(QWidget):
    print("debugging for skull reconstruction system")
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skull Reconstruction")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(AppStyle.MAIN_WINDOW_STYLE)

        # Main container with card-like appearance
        self.main_container = QWidget()
        self.main_container.setStyleSheet(AppStyle.CARD_STYLE)
        self.main_container.setMaximumWidth(500)

        # Create gradient background
        self.setAutoFillBackground(True)
        palette = self.palette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#f5f7fa"))
        gradient.setColorAt(1.0, QColor("#dde4eb"))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)

        # Logo
        self.logo_label = QLabel()
        self.logo_label.setText("🦴")
        self.logo_label.setStyleSheet(f"color: {AppStyle.PRIMARY_COLOR}; font-size: 60px;")
        self.logo_label.setAlignment(Qt.AlignCenter)

        # Title
        self.title_label = QLabel("3D Skull Reconstruction")
        self.title_label.setStyleSheet(
            f"color: {AppStyle.PRIMARY_COLOR}; font-size: 28px; font-weight: bold; {AppStyle.TITLE_FONT}")
        self.title_label.setAlignment(Qt.AlignCenter)

        # Description
        self.description_label = QLabel("Advanced system for cranial implant design")
        self.description_label.setStyleSheet(
            f"color: {AppStyle.TEXT_COLOR}; font-size: 16px; margin-top: 5px; margin-bottom: 25px;")
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setWordWrap(True)

        # Additional description
        self.sub_description = QLabel(
            "Generate customized implants for cranial defects using advanced 3D reconstruction algorithms")
        self.sub_description.setStyleSheet(f"color: {AppStyle.TEXT_COLOR}; font-size: 14px; margin-bottom: 30px;")
        self.sub_description.setAlignment(Qt.AlignCenter)
        self.sub_description.setWordWrap(True)

        # Start Button
        self.start_button = QPushButton("Start Implant Design")
        self.start_button.setStyleSheet(AppStyle.ACTION_BUTTON_STYLE)
        self.start_button.setMinimumHeight(50)
        self.start_button.clicked.connect(self.open_algorithm_selection)

        # Container Layout
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(40, 40, 40, 40)
        container_layout.setSpacing(15)
        container_layout.addWidget(self.logo_label)
        container_layout.addWidget(self.title_label)
        container_layout.addWidget(self.description_label)
        container_layout.addWidget(self.sub_description)
        container_layout.addWidget(self.start_button)
        container_layout.setAlignment(Qt.AlignCenter)
        self.main_container.setLayout(container_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addStretch()
        main_layout.addWidget(self.main_container, 0, Qt.AlignCenter)
        main_layout.addStretch()

        self.setLayout(main_layout)

    def open_algorithm_selection(self):
        self.algorithm_selection = AlgorithmSelection()
        # Check if the current window is maximized
        if self.isMaximized():
            self.algorithm_selection.showMaximized()
        else:
            self.algorithm_selection.show()

        #self.algorithm_selection.show()
        self.close()


class AlgorithmSelection(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Algorithm")
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet(AppStyle.MAIN_WINDOW_STYLE)

        # Main container with card-like appearance
        self.main_container = QWidget()
        self.main_container.setStyleSheet(AppStyle.CARD_STYLE)
        self.main_container.setMaximumWidth(800)

        # Title
        self.title_label = QLabel("Select Reconstruction Method")
        self.title_label.setStyleSheet(
            f"color: {AppStyle.PRIMARY_COLOR}; font-size: 24px; font-weight: bold; {AppStyle.TITLE_FONT} margin-bottom: 10px;")
        self.title_label.setAlignment(Qt.AlignCenter)

        # Subtitle
        self.subtitle = QLabel("Choose the appropriate algorithm for your patient case")
        self.subtitle.setStyleSheet(f"color: {AppStyle.TEXT_COLOR}; font-size: 16px; margin-bottom: 30px;")
        self.subtitle.setAlignment(Qt.AlignCenter)

        # Algorithm cards container
        algorithm_container = QWidget()
        algorithm_layout = QHBoxLayout()
        algorithm_layout.setSpacing(30)

        # Mirror Algorithm Card
        mirror_card = QFrame()
        mirror_card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-radius: {AppStyle.BORDER_RADIUS};
                border: 1px solid {AppStyle.BORDER_COLOR};
            }}
            QFrame:hover {{
                border: 1px solid {AppStyle.SECONDARY_COLOR};
            }}
        """)
        mirror_layout = QVBoxLayout()

        # Mirror icon/image
        mirror_icon = QLabel("🔄")
        mirror_icon.setStyleSheet(f"font-size: 50px; color: {AppStyle.PRIMARY_COLOR}; margin-bottom: 10px;")
        mirror_icon.setAlignment(Qt.AlignCenter)

        # Mirror title
        mirror_title = QLabel("Mirror Imaging")
        mirror_title.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {AppStyle.PRIMARY_COLOR}; {AppStyle.TITLE_FONT}")
        mirror_title.setAlignment(Qt.AlignCenter)

        # Mirror description
        mirror_desc = QLabel(
            "Generate implant by mirroring the healthy side of skull.")
        mirror_desc.setStyleSheet(f"font-size: 15px; color: {AppStyle.TEXT_COLOR}; margin-top: 5px;")
        mirror_desc.setWordWrap(True)
        mirror_desc.setAlignment(Qt.AlignCenter)

        # Mirror button
        self.mirror_button = QPushButton("Select Mirror Method")
        self.mirror_button.setStyleSheet(AppStyle.PRIMARY_BUTTON_STYLE)
        self.mirror_button.clicked.connect(self.start_mirror_imaging)

        mirror_layout.addWidget(mirror_icon)
        mirror_layout.addWidget(mirror_title)
        mirror_layout.addWidget(mirror_desc)
        mirror_layout.addStretch()
        mirror_layout.addWidget(self.mirror_button)
        mirror_card.setLayout(mirror_layout)


        algorithm_layout.addWidget(mirror_card)
        algorithm_container.setLayout(algorithm_layout)

        # Back Button
        self.back_button = QPushButton("Back")
        self.back_button.setStyleSheet(AppStyle.BACK_BUTTON_STYLE)
        self.back_button.clicked.connect(self.go_back)

        # Container Layout
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(40, 40, 40, 40)
        container_layout.addWidget(self.title_label)
        container_layout.addWidget(self.subtitle)
        container_layout.addWidget(algorithm_container)
        container_layout.addSpacing(20)
        container_layout.addWidget(self.back_button, 0, Qt.AlignLeft)
        self.main_container.setLayout(container_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addStretch()
        main_layout.addWidget(self.main_container, 0, Qt.AlignCenter)
        main_layout.addStretch()

        self.setLayout(main_layout)

    def start_mirror_imaging(self):
        self.viewer = STLViewer(algorithm="mirror")
        #self.viewer.show()
        # Check if the current window is maximized
        if self.isMaximized():
            self.viewer.showMaximized()
        else:
            self.viewer.show()
        self.close()



    def go_back(self):
        self.main_window = SkullReconstruction()
        # Check if the current window is maximized
        if self.isMaximized():
            self.main_window.showMaximized()
        else:
            self.main_window.show()
        #self.main_window.show()
        self.close()



### STL Viewer Window ###
class STLViewer(QMainWindow):
    def __init__(self, algorithm="mirror"):
        super().__init__()
        self.setWindowTitle("STL Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.algorithm = algorithm
        self.setStyleSheet(AppStyle.MAIN_WINDOW_STYLE)

        # Create central widget and content area
        central_widget = QWidget()
        self.layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Header section with gradient and more professional look
        header_container = QWidget()
        header_container.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                       stop:0 {AppStyle.PRIMARY_COLOR}, 
                                       stop:1 #0077e6);
            border-radius: {AppStyle.BORDER_RADIUS};
            box-shadow: 0px 4px 10px rgba(0, 102, 204, 0.2);
        """)
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(25, 18, 25, 18)

        # Left side content
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)

        # Title
        self.title_label = QLabel("Skull Implant Generation")
        self.title_label.setStyleSheet("color: white; font-size: 22px; font-weight: bold; letter-spacing: 0.5px;")
        left_layout.addWidget(self.title_label)

        # Subtitle
        self.desc_label = QLabel(f"Using {algorithm.title()} algorithm - Browse and upload file in .stl")
        self.desc_label.setStyleSheet("color: rgba(255, 255, 255, 0.9); font-size: 14px;")
        left_layout.addWidget(self.desc_label)

        header_layout.addWidget(left_content, 1)

        # Right side icon/logo
        icon_container = QWidget()
        icon_container.setFixedSize(50, 50)
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)

        icon_label = QLabel("🦴")
        icon_label.setStyleSheet(
            "color: white; font-size: 30px; background-color: rgba(255, 255, 255, 0.15); border-radius: 25px;")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(icon_label)

        header_layout.addWidget(icon_container)

        self.layout.addWidget(header_container)

        # File Chooser section in a card
        file_section = QWidget()
        file_section.setStyleSheet(AppStyle.CARD_STYLE)
        file_layout = QVBoxLayout(file_section)

        # File browser section title
        file_title = QLabel("Select 3D Model")
        file_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {AppStyle.PRIMARY_COLOR};")
        file_layout.addWidget(file_title)

        # File chooser button row
        button_layout = QHBoxLayout()

        self.load_button = QPushButton("Choose file")
        self.load_button.setFixedWidth(120)
        self.load_button.setStyleSheet(AppStyle.SECONDARY_BUTTON_STYLE)
        self.load_button.clicked.connect(self.load_stl_file)
        button_layout.addWidget(self.load_button)

        self.file_name_label = QLabel("No file selected")
        self.file_name_label.setStyleSheet(
            f"color: {AppStyle.TEXT_COLOR}; padding: 5px 10px; border: 1px dashed {AppStyle.BORDER_COLOR}; border-radius: 4px;")
        button_layout.addWidget(self.file_name_label, 1)

        file_layout.addLayout(button_layout)
        self.layout.addWidget(file_section)

        # Model viewer section
        viewer_section = QWidget()
        viewer_section.setStyleSheet(AppStyle.CARD_STYLE)
        viewer_layout = QVBoxLayout(viewer_section)

        # View 3D Model Label
        self.view_model_label = QLabel("View the 3D model")
        self.view_model_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {AppStyle.PRIMARY_COLOR};")
        viewer_layout.addWidget(self.view_model_label)

        # VTK Render Window (in a frame for better appearance)
        render_frame = QFrame()
        render_frame.setFrameShape(QFrame.StyledPanel)
        render_frame.setStyleSheet(
            f"border: 1px solid {AppStyle.BORDER_COLOR}; border-radius: {AppStyle.BORDER_RADIUS};")
        render_layout = QVBoxLayout(render_frame)

        self.render_window = QVTKRenderWindowInteractor(self)
        render_layout.addWidget(self.render_window)

        viewer_layout.addWidget(render_frame)
        self.layout.addWidget(viewer_section)

        # Bottom Row Buttons
        button_container = QWidget()
        button_row = QHBoxLayout(button_container)

        # Back Button
        self.back_button = QPushButton("Back")
        self.back_button.setStyleSheet(AppStyle.BACK_BUTTON_STYLE)
        self.back_button.clicked.connect(self.go_back)
        button_row.addWidget(self.back_button)

        button_row.addStretch()

        # Next Button
        self.next_button = QPushButton("Next")
        self.next_button.setStyleSheet(AppStyle.PRIMARY_BUTTON_STYLE)
        self.next_button.clicked.connect(self.next)
        self.next_button.setEnabled(False)  # Disable until file is loaded
        button_row.addWidget(self.next_button)

        self.layout.addWidget(button_container)

        # VTK Renderer
        self.renderer = vtk.vtkRenderer()
        self.render_window.GetRenderWindow().AddRenderer(self.renderer)
        self.reader = None  # To store the STL file reader

        # Initialize the interactor
        self.iren = self.render_window.GetRenderWindow().GetInteractor()
        self.iren.Initialize()


    virus=0
    def load_stl_file(self):
        """Open a file dialog to choose an STL file and ensure it is ≤ 400MB."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open STL File", "", "STL Files (*.stl)")

        if file_name:
            file_path = file_name

            def calculate_hash(self, file_path):

                """Compute SHA256 hash of the STL file."""
                hasher = hashlib.sha256()
                with open(file_path, "rb") as f:
                    while chunk := f.read(4096):
                        hasher.update(chunk)
                return hasher.hexdigest()


            def scan_file(self, file_path):
                """Check the file hash against a malware database."""
                file_hash = calculate_hash( self,file_path)
                wrapper = "https://mb-api.abuse.ch/api/v1/"
                response = requests.post(wrapper, data={"query": "get_info", "hash": file_hash})

                if "malware" in response.text:
                    print("the file contains virus")
                    QMessageBox.critical(self, "Security Warning", "❌ Virus detected! File loading blocked.")
                    return False
                self.virus=1
                print("the stl file is virus free")
                return True

            scan_file(self, file_path)
            try:
                file_size_mb = os.path.getsize(file_name) / (1024 * 1024)  # Convert bytes to MB
                if file_size_mb > 600:
                    QMessageBox.warning(self, "File Size Error",
                                        "The selected STL file exceeds 400MB. Please choose a smaller file.")
                    print("file size exceeds limit")
                    return

                self.file_name_label.setText(os.path.basename(file_name))
                self.file_name_label.setStyleSheet(f"""
                    color: {AppStyle.PRIMARY_COLOR}; 
                    padding: 5px 10px; 
                    border: 1px solid {AppStyle.SECONDARY_COLOR}; 
                    border-radius: 4px;
                    background-color: rgba(120, 197, 214, 0.1);
                """)
                self.stl_file_path = file_name  # Store the file path

                # Read STL File
                self.reader = vtk.vtkSTLReader()
                self.reader.SetFileName(file_name)
                self.reader.Update()

                # Create a mapper and actor for STL file
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(self.reader.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                # Clear previous actors
                self.renderer.RemoveAllViewProps()

                # Add actor to renderer
                self.renderer.AddActor(actor)
                self.renderer.SetBackground(0.1, 0.1, 0.1)
                self.renderer.ResetCamera()

                # Start rendering
                self.render_window.GetRenderWindow().Render()
                self.iren.Start()

                # Enable the next button
                self.next_button.setEnabled(True)


            except Exception as e:
                self.show_error_message(f"Error loading STL file: {str(e)}")




    def next(self):
        """Split the skull model into two equal halves using PyVista."""
        if not self.reader or self.virus == 0:
            self.show_error_message("No STL file loaded.")
            return

        try:

            # Show a progress dialog
            progress_container = QWidget()
            progress_container.setStyleSheet(AppStyle.CARD_STYLE)
            progress_layout = QVBoxLayout(progress_container)

            progress_title = QLabel("Processing Model")
            progress_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {AppStyle.PRIMARY_COLOR};")
            progress_layout.addWidget(progress_title)

            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setTextVisible(True)
            progress_layout.addWidget(progress)

            self.layout.addWidget(progress_container)

            # Update progress
            progress.setValue(10)
            QApplication.processEvents()

            # Convert VTK polydata to PyVista mesh
            skull_vtk = self.reader.GetOutput()
            skull = pv.wrap(skull_vtk)

            progress.setValue(30)
            QApplication.processEvents()

            # Get the bounds of the model to find the midpoint along the X-axis
            bounds = skull.bounds
            mid_x = (bounds[0] + bounds[1]) / 2
            #print("Model Bounds:", bounds)
            print("Midpoint X:", mid_x)

            # Create a plane at the midpoint along the X-axis
            plane = pv.Plane(
                center=(mid_x, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2),
                direction=(1, 0, 0),
                i_size=(bounds[3] - bounds[2]) * 1.5,
                j_size=(bounds[5] - bounds[4]) * 1.5,
            )

            progress.setValue(50)
            QApplication.processEvents()

            # Clip the skull using the plane
            left_half = skull.clip_surface(plane, invert=False)
            right_half = skull.clip_surface(plane, invert=True)

            progress.setValue(70)
            QApplication.processEvents()

            # Check voxel counts and save the half with more voxels as undamaged
            left_voxel_count = left_half.volume
            right_voxel_count = right_half.volume

            print(f"Voxel count - Left half: {left_voxel_count}")
            print(f"Voxel count - Right half: {right_voxel_count}")

            # Create a temporary directory to store the split meshes
            temp_dir = os.path.join(os.getcwd(), "temp")
            os.makedirs(temp_dir, exist_ok=True)

            if left_voxel_count >= right_voxel_count:
                undamaged_half = left_half
                damaged_half = right_half
                undamaged_side = "left"
                left_half.save(os.path.join(temp_dir, "undamaged_half.stl"))
                print("Undamaged half saved as 'undamaged_half.stl'.")
                if right_voxel_count > 0:
                    right_half.save(os.path.join(temp_dir, "damaged_half.stl"))
                    print("Damaged half saved as 'damaged_half.stl'.")
                else:
                    print("Warning: Right half is empty (damaged).")
            else:
                undamaged_half = right_half
                damaged_half = left_half
                undamaged_side = "right"
                right_half.save(os.path.join(temp_dir, "undamaged_half.stl"))
                print("Undamaged half saved as 'undamaged_half.stl'.")
                if left_voxel_count > 0:
                    left_half.save(os.path.join(temp_dir, "damaged_half.stl"))
                    print("Damaged half saved as 'damaged_half.stl'.")
                else:
                    print("Warning: Left half is empty (damaged).")

            undamaged_path1 = os.path.join(temp_dir, "undamaged_half.stl")
            # Load the STL file
            mesh = pv.read(undamaged_path1)

            # Create a 4x4 transformation matrix (identity matrix)
            transformation_matrix = np.eye(4)

            # Set the mirroring transformation
            axis = 'x'
            if axis == 'x':
                transformation_matrix[0, 0] = -1  # Mirror along X-axis
            elif axis == 'y':
                transformation_matrix[1, 1] = -1  # Mirror along Y-axis
            elif axis == 'z':
                transformation_matrix[2, 2] = -1  # Mirror along Z-axis
            else:
                raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

            # Apply the transformation to mirror the model
            mirrored_mesh = mesh.transform(transformation_matrix)
            k = mirrored_mesh

            # Save the mirrored mesh to a new STL file
            k.save(os.path.join(temp_dir, "mirrored_x.stl"))
            print("mirrored image saved as 'mirrored_x.stl'.")

            progress.setValue(100)
            QApplication.processEvents()

            # Remove the progress bar
            self.layout.removeWidget(progress_container)
            progress_container.deleteLater()

            # Get camera information for the next view
            camera = self.renderer.GetActiveCamera()
            camera_position = camera.GetPosition()
            focal_point = camera.GetFocalPoint()
            view_up = camera.GetViewUp()

            camera_info = {
                "position": camera_position,
                "focal_point": focal_point,
                "view_up": view_up
            }

            # Open the damaged model view
            damaged_path = os.path.join(temp_dir, "damaged_half.stl")
            undamaged_path = os.path.join(temp_dir, "undamaged_half.stl")
            self.open_damaged_model_view(damaged_path, undamaged_path, camera_info, undamaged_side)

        except Exception as e:
            self.show_error_message(f"Error processing skull model: {str(e)}")
            print(f"error:{str(e)}")

    def open_damaged_model_view(self, damaged_path, undamaged_path, camera_info, undamaged_side):
        """Open a new window to display the damaged half STL model."""
        try:
            self.damaged_model_view = DamagedModelView(damaged_path, undamaged_path, camera_info, undamaged_side,
                                                       self.algorithm)
            #self.damaged_model_view.show()
            if self.isMaximized():
                self.damaged_model_view.showMaximized()
            else:
                self.damaged_model_view.show()
            self.close()
        except Exception as e:
            self.show_error_message(f"Error opening damaged model view: {str(e)}")
            print(f"error:{str(e)}")


    def go_back(self):
        """Return to the algorithm selection screen."""
        self.main_window = AlgorithmSelection()
        #self.main_window.show()
        if self.isMaximized():
            self.main_window.showMaximized()
        else:
            self.main_window.show()

        self.close()

    def show_error_message(self, message):
        """Display an error message dialog."""
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.setStyleSheet(f"QLabel {{ min-width: 300px; }}")
        error_dialog.exec_()


### Damaged Model View ###
### Damaged Model View ###
#tryy
class DamagedModelView(QMainWindow):
    def __init__(self, damaged_path, undamaged_path, camera_info, undamaged_side, algorithm):
        super().__init__()
        self.damaged_path = damaged_path
        self.undamaged_path = undamaged_path
        self.camera_info = camera_info
        self.undamaged_side = undamaged_side
        self.algorithm = algorithm

        self.setWindowTitle("Damaged Model Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(AppStyle.MAIN_WINDOW_STYLE)
        # Main Layout
        self.layout = QVBoxLayout()

        # Header with gradient
        header_container = QWidget()
        header_container = QWidget()
        header_container.setStyleSheet(f"""
                            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                       stop:0 {AppStyle.PRIMARY_COLOR}, 
                                                       stop:1 #0077e6);
                            border-radius: {AppStyle.BORDER_RADIUS};
                            box-shadow: 0px 4px 10px rgba(0, 102, 204, 0.2);
                        """)
        header_layout = QHBoxLayout(header_container)

        # Left content with title and icon
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)

        self.title_label = QLabel("Set Position And Thickness")
        self.title_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        title_layout.addWidget(self.title_label)

        # Right side icon/logo
        icon_container = QWidget()
        icon_container.setFixedSize(50, 50)
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)

        icon_label = QLabel("🔍")
        icon_label.setStyleSheet(
            "color: white; font-size: 30px; background-color: rgba(255, 255, 255, 0.15); border-radius: 25px;")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(icon_label)

        header_layout.addWidget(title_container)
        header_layout.addWidget(icon_container)
        self.layout.addWidget(header_container)

        # Description
        self.desc_label = QLabel(f"press d to mark, a to remove marking and s to set position")
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.desc_label)

        # VTK Render Window
        self.render_window = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.render_window)

        # Bottom Row Buttons
        button_row = QHBoxLayout()

        # Back Button
        self.back_button = QPushButton("← Back")
        self.back_button.setStyleSheet(AppStyle.BACK_BUTTON_STYLE)
        self.back_button.clicked.connect(self.go_back)
        button_row.addWidget(self.back_button)
        button_row.addStretch()

        # Next Button
        self.next_button = QPushButton("Detect Damage ⚡")
        self.next_button.setStyleSheet(AppStyle.PRIMARY_BUTTON_STYLE)
        self.next_button.clicked.connect(self.on_next_button_clicked)
        button_row.addWidget(self.next_button)

        self.layout.addLayout(button_row)
        # Generate Report button
        report_button = QPushButton("customize")
        report_button.setStyleSheet(AppStyle.SECONDARY_BUTTON_STYLE)
        report_button.clicked.connect(self.customization)
        button_row.addWidget(report_button)

        button_row.addLayout(button_row)

        # Container widget
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # VTK Renderer
        self.renderer = vtk.vtkRenderer()
        self.render_window.GetRenderWindow().AddRenderer(self.renderer)

        # Initialize Lists for Points
        self.dots = []
        self.point_actors = []

        # Load the STL File
        self.load_damaged_half()

        # Initialize the interactor
        self.iren = self.render_window.GetRenderWindow().GetInteractor()
        self.apply_interactor_style()
        self.iren.Initialize()

    def load_damaged_half(self):
        self.edge_detection_value=8
        self.damage_size_value=4000
        """Load and display the damaged STL model."""
        try:
            reader = vtk.vtkSTLReader()
            reader.SetFileName(self.damaged_path)
            reader.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            self.actor = vtk.vtkActor()
            self.actor.SetMapper(mapper)
            self.actor.GetProperty().SetColor(1.0, 0.5, 0.5)  # Red color for damaged

            self.renderer.RemoveAllViewProps()
            self.renderer.AddActor(self.actor)
            self.renderer.SetBackground(0.1, 0.1, 0.1)
            print("damaged half has been loaded")


            if self.camera_info:
                camera = self.renderer.GetActiveCamera()
                camera.SetPosition(self.camera_info["position"])
                camera.SetFocalPoint(self.camera_info["focal_point"])
                camera.SetViewUp(self.camera_info["view_up"])
            else:
                self.renderer.ResetCamera()

            self.render_window.GetRenderWindow().Render()

        except Exception as e:
            self.show_error_message(f"Error loading damaged half: {str(e)}")

    def apply_interactor_style(self):
        """Applies the custom interactor style for point selection."""
        class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
                self.AddObserver("KeyPressEvent", self.on_key_press)

            def on_key_press(self, obj, event):
                key = self.GetInteractor().GetKeySym()

                if key == "d":  # Select a point
                    click_pos = self.GetInteractor().GetEventPosition()
                    picker = vtk.vtkCellPicker()
                    picker.SetTolerance(0.005)
                    picker.Pick(click_pos[0], click_pos[1], 0, self.parent.renderer)
                    picked_position = picker.GetPickPosition()

                    if picked_position != (0.0, 0.0, 0.0):
                        self.parent.add_point(picked_position)

                elif key == "a":  # Remove last point
                    self.parent.remove_last_point()

                elif key == "s":  # Print distance
                    distance = self.parent.calculate_distance()
                    if distance is not None:
                        print(f"thickness of skull: {distance:.2f} mm")
                        msg_box = QMessageBox()
                        msg_box.setIcon(QMessageBox.Information)
                        msg_box.setWindowTitle("Distance")
                        msg_box.setText(f"Distance between points: {distance:.2f} mm")
                        msg_box.exec_()
                    else:
                        print("Not enough points to calculate distance.")
                        msg_box = QMessageBox()
                        msg_box.setIcon(QMessageBox.Warning)
                        msg_box.setWindowTitle("Error")
                        msg_box.setText("Not enough points to calculate distance")
                        msg_box.exec_()

                self.GetInteractor().GetRenderWindow().Render()

        self.iren.SetInteractorStyle(CustomInteractorStyle(self))

    def add_point(self, position):
        """Add a red dot at the clicked 3D position."""
        if len(self.dots) >= 2:
            self.dots.pop(0)  # Keep only 2 points
            self.renderer.RemoveActor(self.point_actors.pop(0))

        self.dots.append(position)

        # Create a sphere to represent the dot
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(position)
        sphere.SetRadius(1)

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())

        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(1, 0, 0)  # Red color

        self.renderer.AddActor(sphere_actor)
        self.point_actors.append(sphere_actor)

        self.render_window.GetRenderWindow().Render()

    def remove_last_point(self):
        """Remove the last added point."""
        if self.dots:
            self.dots.pop()
            self.renderer.RemoveActor(self.point_actors.pop())
            self.render_window.GetRenderWindow().Render()

    def calculate_distance(self):
        """Calculate and return the Euclidean distance between two points in mm."""
        if len(self.dots) == 2:
            pos1 = np.array(self.dots[0])
            pos2 = np.array(self.dots[1])
            self.thickness=np.linalg.norm(pos2 - pos1)
            na=self.thickness
            return na
        return None
    #try
    def customization(self):
        # Create and style the report dialog
        report_dialog = QDialog(self)
        report_dialog.setWindowTitle("Damage detection sensitivity")
        report_dialog.setMinimumWidth(500)
        report_dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {AppStyle.BACKGROUND_COLOR};
                border-radius: {AppStyle.BORDER_RADIUS};
            }}
            QLabel {{
                color: {AppStyle.TEXT_COLOR};
                padding: 5px;
            }}
            QGroupBox {{
                border: 1px solid {AppStyle.BORDER_COLOR};
                border-radius: {AppStyle.BORDER_RADIUS};
                margin-top: 10px;
                background-color: {AppStyle.CARD_COLOR};
            }}
            QGroupBox::title {{
                color: {AppStyle.PRIMARY_COLOR};
                padding: 0 10px;
            }}
        """)

        # Create layout
        layout = QVBoxLayout(report_dialog)
        layout.setSpacing(15)

        # Header
        header = QLabel("Set Sensitivity")
        header.setStyleSheet(f"""
            font-size: 20px;
            font-weight: bold;
            color: {AppStyle.PRIMARY_COLOR};
            padding: 10px;
            border-bottom: 2px solid {AppStyle.BORDER_COLOR};
        """)
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Information groups
        info_group = QGroupBox()
        info_layout = QVBoxLayout()

        # Edge Detection Sensitivity Bar
        self.sensitivity_bar = QSlider(Qt.Horizontal)
        self.sensitivity_bar.setRange(1, 20)
        self.sensitivity_bar.setValue(8)
        self.sensitivity_label = QLabel(f"Edge Detection Sensitivity: {self.sensitivity_bar.value()}")
        self.sensitivity_bar.valueChanged.connect(
            lambda: self.sensitivity_label.setText(f"Edge Detection Sensitivity: {self.sensitivity_bar.value()}"))

        # Minimum Damage Size Bar
        self.damage_size_bar = QSlider(Qt.Horizontal)
        self.damage_size_bar.setRange(1, 100000)
        self.damage_size_bar.setValue(4000)
        self.damage_size_label = QLabel(f"Minimum Damage Size: {self.damage_size_bar.value()}")
        self.damage_size_bar.valueChanged.connect(
            lambda: self.damage_size_label.setText(f"Minimum Damage Size: {self.damage_size_bar.value()}"))

        # Adding sliders and labels in the same format
        for label, bar, bar_label in [
            ("Edge Detection Sensitivity", self.sensitivity_bar, self.sensitivity_label),
            ("Minimum Damage Size", self.damage_size_bar, self.damage_size_label)
        ]:
            detail_widget = QWidget()
            detail_layout = QHBoxLayout(detail_widget)

            label_widget = QLabel(f"{label}:")
            label_widget.setStyleSheet(f"color: {AppStyle.TEXT_COLOR}; font-weight: bold;")

            bar_layout = QVBoxLayout()
            bar_layout.addWidget(bar_label)
            bar_layout.addWidget(bar)

            detail_layout.addWidget(label_widget)
            detail_layout.addLayout(bar_layout)
            detail_layout.addStretch()

            info_layout.addWidget(detail_widget)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Buttons
        button_layout = QHBoxLayout()

        save_button = QPushButton("Apply Changes")
        save_button.setStyleSheet(AppStyle.PRIMARY_BUTTON_STYLE)
        save_button.clicked.connect(self.save_info)

        close_button = QPushButton("Close")
        close_button.setStyleSheet(AppStyle.SECONDARY_BUTTON_STYLE)
        close_button.clicked.connect(report_dialog.close)

        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        report_dialog.exec_()

    def save_info(self):
        self.edge_detection_value = self.sensitivity_bar.value()
        self.damage_size_value = self.damage_size_bar.value()
        #print(f"Saved Values - Edge Detection: {self.edge_detection_value}, Damage Size: {self.damage_size_value}")
        # Show a notification message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Success")
        msg_box.setText("Changes have been saved successfully!")
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    def on_next_button_clicked(self):
        """Generate depth image and detect damaged areas"""
        # Show a progress dialog
        print(f"Saved Values - Edge Detection: {self.edge_detection_value}, Damage Size: {self.damage_size_value}")
        progress = QProgressBar(self)
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setTextVisible(True)
        self.layout.addWidget(progress)
        QApplication.processEvents()


        try:
            progress.setValue(10)
            QApplication.processEvents()
            fg=self.edge_detection_value
            fu=self.damage_size_value

            # Get the camera information
            camera = self.renderer.GetActiveCamera()
            camera_position = camera.GetPosition()
            focal_point = camera.GetFocalPoint()
            view_up = camera.GetViewUp()
            self.camera_info['position'] = camera_position
            self.camera_info['focal_point'] = focal_point
            self.camera_info['view_up'] = view_up
            print("camera position:",camera_position)
            print("focal point:",focal_point)

            print("view up:",view_up)


            # Load the damaged half in PyVista
            damaged_mesh = pv.read(self.damaged_path)

            progress.setValue(20)
            QApplication.processEvents()

            # Initialize a PyVista plotter for depth map generation
            plotter_pyvista = pv.Plotter(off_screen=True)
            plotter_pyvista.add_mesh(damaged_mesh)

            # Set the camera position correctly
            plotter_pyvista.camera_position = [camera_position, focal_point, view_up]

            progress.setValue(30)
            QApplication.processEvents()

            # Fix clipping range
            plotter_pyvista.camera.clipping_range = (0.1, 5000)

            # Ensure rendering before depth extraction
            plotter_pyvista.show(auto_close=False)
            plotter_pyvista.render()

            progress.setValue(40)
            QApplication.processEvents()

            # Capture depth values
            depth_values = plotter_pyvista.get_image_depth()

            # Check if depth values are valid
            if depth_values is not None:
                # Fix negative depth values
                depth_values = np.abs(depth_values)  # Convert all values to positive

                # Replace NaN values with 0
                depth_values = np.nan_to_num(depth_values, nan=0)
            print("depthmap generated for damaged half using the position user set before")

            progress.setValue(50)
            QApplication.processEvents()

            # Apply edge detection (Canny)
            edges = feature.canny(depth_values, sigma=fg)
            print("edge detection has been applied")

            # Apply dilation on the detected edges
            dilated_edges = morphology.dilation(edges, morphology.disk(5))
            print("dilation has been appliead to fill small holes in the detected edge to ensure continuity")

            progress.setValue(60)
            QApplication.processEvents()

            # Label connected regions in the dilated edge-detected image
            labels = measure.label(dilated_edges)

            # Create a mask to exclude the largest region (outer edge) and non-ring (open) regions
            mask = np.ones_like(dilated_edges, dtype=bool)

            progress.setValue(70)
            QApplication.processEvents()

            # Iterate over all regions
            for region in measure.regionprops(labels):
                # Check if the region is the largest (outer edge)
                if region == max(measure.regionprops(labels), key=lambda r: r.area):
                    # Remove the outer edge (largest region)
                    for coord in region.coords:
                        mask[coord[0], coord[1]] = False
                # Check if the region forms a closed ring using the Euler number (<= 0 means closed/ring-like)
                elif region.euler_number <= 0 and region.area >= fu:
                    continue  # Keep this region as it forms a ring and has enough area
                else:
                    # Remove open regions that do not form a ring
                    for coord in region.coords:
                        mask[coord[0], coord[1]] = False

            # Apply the mask to remove the outer edge, non-ring regions, and small rings
            internal_edges = np.logical_and(dilated_edges, mask)

            progress.setValue(80)
            QApplication.processEvents()
            # Check if any edges remain after filtering
            if not np.any(internal_edges):
                #raise ValueError("No edge found after applying filtering logic.")
                self.show_error_message(f"no edge found please set position correctly or upload a valid damaged model")
            else:
                # Create an RGB image from the original grayscale depth image
                colored_image = np.stack([depth_values] * 3, axis=-1)  # Convert grayscale depth to 3-channel RGB
                # Define the color for the edges (e.g., red)
                edge_color = [1, 0, 0]  # Red color

                # Color the edges in the RGB image
                colored_image[internal_edges] = edge_color  # Assign red to the edge pixels
                # Calculate the perimeter of the red ring
                red_ring_perimeter_pixels = np.sum(internal_edges)  # Count the edge pixels
                # Convert pixel perimeter to real-world units if needed
                red_ring_perimeter = red_ring_perimeter_pixels * 0.01  # Example: 0.01 cm per pixel
                print("detected damage size=",red_ring_perimeter)


                # 1. Create a mask from the highlighted edges in red
                filled_mask = internal_edges.copy()  # Start with the detected ring edges

                # 2. Use binary_fill_holes to fill enclosed areas within the ring
                filled_mask = binary_fill_holes(filled_mask)  # Fill the enclosed regions

                # 3. Overlay the filled region onto the original colored image
                # Define the fill color (e.g., red)
                fill_color = [1, 0, 0]  # Red color

                # Apply the fill color to the filled areas in the RGB image
                colored_image[filled_mask] = fill_color

                # Save x, y coordinates of red pixels in the filled ring to a CSV file
                red_pixels = np.argwhere(filled_mask)  # Get coordinates of filled ring pixels
                # Ensure colored_image is in the range [0,1]
                colored_image = np.clip(colored_image, 0, 1)  # Clip values to valid range

                progress.setValue(90)
                QApplication.processEvents()

                # Save the colored edges image to a temporary file
                temp_dir = os.path.join(os.getcwd(), "temp")
                os.makedirs(temp_dir, exist_ok=True)
                colored_image_path = os.path.join(temp_dir, "colored_image.png")

                # Ensure the image data is properly scaled for saving
                # plt.imsave(colored_image_path, colored_image)
                if colored_image is None:
                    print("Error: colored_image is None!")

                elif not np.any(colored_image):
                    print("Error: colored_image is completely black or empty!")

                elif not os.path.exists(os.path.dirname(colored_image_path)):
                    print("Error: Directory does not exist!")

                else:
                    plt.imsave(colored_image_path, colored_image)
                    print(f"Image saved successfully at {colored_image_path}")

                progress.setValue(100)
                QApplication.processEvents()
                thickness = self.thickness
                # Proceed to the detected damage view
                self.show_detected_damage(colored_image_path, red_ring_perimeter, thickness,fg,fu)

                # Remove the progress bar
            self.layout.removeWidget(progress)
            progress.deleteLater()


        except Exception as e:
            # Remove the progress bar
            self.layout.removeWidget(progress)
            progress.deleteLater()

            self.show_error_message(f"Error detecting damage: {str(e)}")
            print(f"error:{str(e)}")


    def show_detected_damage(self, image_path, perimeter,thickness,fg,fu):
        """Open the detected damage view window"""
        try:
            self.detected_damage_view = DetectedDamageView(image_path, perimeter,thickness,fg,fu,
                                                           self.damaged_path, self.undamaged_path,
                                                           self.camera_info, self.undamaged_side,
                                                           self.algorithm)
            #self.detected_damage_view.show()
            if self.isMaximized():
                self.detected_damage_view.showMaximized()
            else:
                self.detected_damage_view.show()
            self.close()
        except Exception as e:
            self.show_error_message(f"Error showing detected damage: {str(e)}")
            print(f"error:{str(e)}")


    def go_back(self):
        """Return to the algorithm selection screen."""
        self.main_window = STLViewer()
        # self.main_window.show()
        if self.isMaximized():
            self.main_window.showMaximized()
        else:
            self.main_window.show()

        self.close()

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_()
### Detected Damage View ###
class DetectedDamageView(QMainWindow):
    def __init__(self, colored_image_path, perimeter,thickness,fg,fu, damaged_path, undamaged_path, camera_info, undamaged_side,
                 algorithm):
        super().__init__()
        self.colored_image_path = colored_image_path
        self.perimeter = perimeter
        self.thickness=thickness
        self.fg=fg
        self.fu=fu
        self.damaged_path = damaged_path
        self.undamaged_path = undamaged_path
        self.camera_info = camera_info
        self.undamaged_side = undamaged_side
        self.algorithm = algorithm

        self.setWindowTitle("Detected Damage")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(AppStyle.MAIN_WINDOW_STYLE)

        self.layout = QVBoxLayout()
        # Header Section with Gradient
        header_container = QWidget()
        header_container.setStyleSheet(f"""
                            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                       stop:0 {AppStyle.PRIMARY_COLOR}, 
                                                       stop:1 #0077e6);
                            border-radius: {AppStyle.BORDER_RADIUS};
                            box-shadow: 0px 4px 10px rgba(0, 102, 204, 0.2);
                        """)
        header_layout = QHBoxLayout(header_container)

        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)

        self.title_label = QLabel("Damage Analysis Results")
        self.title_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        title_layout.addWidget(self.title_label)

        icon_container = QWidget()
        icon_container.setFixedSize(50, 50)
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_label = QLabel("🔍")
        icon_label.setStyleSheet(
            "color: white; font-size: 30px; background-color: rgba(255, 255, 255, 0.15); border-radius: 25px;")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(icon_label)

        header_layout.addWidget(title_container)
        header_layout.addWidget(icon_container)

        self.layout.addWidget(header_container)
        # Show the damage image
        self.image_label = QLabel()
        pixmap = QPixmap(colored_image_path)
        self.image_label.setPixmap(pixmap.scaled(600, 400, Qt.KeepAspectRatio))
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # Thickness measurement section
        thickness_group = QGroupBox("Implant Thickness")
        thickness_layout = QVBoxLayout()

        thickness_desc = QLabel("Specify the thickness for the implant (mm):")
        thickness_layout.addWidget(thickness_desc)

        thickness_input_layout = QHBoxLayout()
        self.thickness_spinbox = QDoubleSpinBox()
        self.thickness_spinbox.setRange(0.1, 20.0)
        self.thickness_spinbox.setSingleStep(0.1)
        self.thickness_spinbox.setValue(thickness)  # Default thickness
        thickness_input_layout.addWidget(self.thickness_spinbox)

        thickness_layout.addLayout(thickness_input_layout)
        thickness_group.setLayout(thickness_layout)
        self.layout.addWidget(thickness_group)

        # Bottom Row Buttons
        button_row = QHBoxLayout()

        # Back Button
        self.back_button = QPushButton("← Back")
        self.back_button.setStyleSheet(AppStyle.BACK_BUTTON_STYLE)
        self.back_button.clicked.connect(self.go_back)
        button_row.addWidget(self.back_button)

        button_row.addStretch()

        # Next Button
        self.generate_button = QPushButton("Generate Implant")
        self.generate_button.setStyleSheet(AppStyle.PRIMARY_BUTTON_STYLE)
        self.generate_button.clicked.connect(self.on_generate_implant_clicked)
        button_row.addWidget(self.generate_button)

        self.layout.addLayout(button_row)

        # Container widget
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def on_generate_implant_clicked(self):
        """Generate the implant using the specified thickness and point cloud approach"""
        thickness = self.thickness_spinbox.value()

        # Show a progress dialog
        progress = QProgressBar(self)
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setTextVisible(True)
        self.layout.addWidget(progress)
        QApplication.processEvents()

        try:
            progress.setValue(10)
            QApplication.processEvents()

            # Create directory structure
            temp_dir = os.path.join(os.getcwd(), "temp")
            mirror_dir = os.path.join(os.getcwd(), "MIRROR_IMAGING")
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(mirror_dir, exist_ok=True)

            # Load the damaged half for processing
            damaged_mesh = pv.read(self.damaged_path)

            progress.setValue(20)
            QApplication.processEvents()

            # Get the camera position information
            # try
            # camera = self.renderer.GetActiveCamera()
            # camera_position = camera.GetPosition()
            # focal_point = camera.GetFocalPoint()
            # view_up = camera.GetViewUp()

            # actual
            camera_position = self.camera_info["position"]
            focal_point = self.camera_info["focal_point"]
            view_up = self.camera_info["view_up"]
            #print(camera_position, focal_point, view_up)
            # Initialize a PyVista plotter for depth map generation
            plotter_pyvista = pv.Plotter(off_screen=True)
            plotter_pyvista.add_mesh(damaged_mesh)

            # Set the camera position correctly
            plotter_pyvista.camera_position = [camera_position, focal_point, view_up]

            # Fix clipping range
            plotter_pyvista.camera.clipping_range = (0.1, 5000)

            # Ensure rendering before depth extraction
            plotter_pyvista.show(auto_close=False)
            plotter_pyvista.render()

            progress.setValue(30)
            QApplication.processEvents()

            # Capture depth values
            depth_values = plotter_pyvista.get_image_depth()

            # Check if depth values are valid
            if depth_values is not None:
                # Fix negative depth values
                depth_values = np.abs(depth_values)  # Convert all values to positive

                # Replace NaN values with 0
                depth_values = np.nan_to_num(depth_values, nan=0)

                progress.setValue(40)
                QApplication.processEvents()
                # Visualize the depth map
                #plt.imshow(depth_values, cmap='hot')
                #plt.title('Depth Map')
                #plt.colorbar()
                #plt.axis('off')
                #plt.show()
                edge_sensitivity=self.fg
                size_edge=self.fu

                # Apply edge detection (Canny)
                edges = feature.canny(depth_values, sigma=edge_sensitivity)

                # Apply dilation on the detected edges
                dilated_edges = morphology.dilation(edges, morphology.disk(5))

                progress.setValue(50)
                QApplication.processEvents()

                # Label connected regions in the dilated edge-detected image
                labels = measure.label(dilated_edges)

                # Create a mask to exclude the largest region (outer edge) and non-ring (open) regions
                mask = np.ones_like(dilated_edges, dtype=bool)
                # Iterate over all regions
                for region in measure.regionprops(labels):
                    # Check if the region is the largest (outer edge)
                    if region == max(measure.regionprops(labels), key=lambda r: r.area):
                        # Remove the outer edge (largest region)
                        for coord in region.coords:
                            mask[coord[0], coord[1]] = False
                    # Check if the region forms a closed ring using the Euler number (<= 0 means closed/ring-like)
                    elif region.euler_number <= 0 and region.area >= size_edge:
                        continue  # Keep this region as it forms a ring and has enough area
                    else:
                        # Remove open regions that do not form a ring
                        for coord in region.coords:
                            mask[coord[0], coord[1]] = False

                # Apply the mask to remove the outer edge, non-ring regions, and small rings
                internal_edges = np.logical_and(dilated_edges, mask)

                progress.setValue(60)
                QApplication.processEvents()

                # Create an RGB image from the original grayscale depth image
                colored_image = np.stack([depth_values] * 3, axis=-1)  # Convert grayscale depth to 3-channel RGB

                # Define the color for the edges (e.g., red)
                edge_color = [1, 0, 0]  # Red color

                # Color the edges in the RGB image
                colored_image[internal_edges] = edge_color  # Assign red to the edge pixels

                # 1. Create a mask from the highlighted edges in red
                filled_mask = internal_edges.copy()  # Start with the detected ring edges

                # 2. Use binary_fill_holes to fill enclosed areas within the ring
                filled_mask = binary_fill_holes(filled_mask)  # Fill the enclosed regions

                # 3. Overlay the filled region onto the original colored image
                # Define the fill color (e.g., red)
                fill_color = [1, 0, 0]  # Red color

                # Apply the fill color to the filled areas in the RGB image
                colored_image[filled_mask] = fill_color

                # Save x, y coordinates of red pixels in the filled ring to a CSV file
                red_pixels = np.argwhere(filled_mask)  # Get coordinates of filled ring pixels
                #Visualize the result with the filled ring
                plt.imshow(colored_image)
                plt.title('Filled Red Rings in Depth Image')
                plt.axis('off')
                #plt.show()

                csv_file_path = "red_pixels_coordinates.csv"
                with open(csv_file_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(['x', 'y'])  # Write header
                    for y, x in red_pixels:
                        csv_writer.writerow([x, y])  # Write x, y coordinates of each red pixel
                    print("x,y coordinates of damage saved in to csv file named red_pixels_coordinates.csv")

                # mirrorimage for z
                # Load the damaged half for processing
                mir = os.path.join(temp_dir, "mirrored_x.stl")
                damaged_mesh1 = pv.read(mir)
                # actual
                camera_position = self.camera_info["position"]
                focal_point = self.camera_info["focal_point"]
                view_up = self.camera_info["view_up"]
                #print(camera_position, focal_point, view_up)
                # Initialize a PyVista plotter for depth map generation
                plotter_pyvista = pv.Plotter(off_screen=True)
                plotter_pyvista.add_mesh(damaged_mesh1)
                # Set the camera position correctly
                plotter_pyvista.camera_position = [camera_position, focal_point, view_up]

                # Fix clipping range
                plotter_pyvista.camera.clipping_range = (0.1, 5000)

                # Ensure rendering before depth extraction
                plotter_pyvista.show(auto_close=False)
                plotter_pyvista.render()
                # Capture depth values
                depth_values = plotter_pyvista.get_image_depth()

                # Check if depth values are valid
                if depth_values is not None:
                    # Fix negative depth values
                    depth_values = np.abs(depth_values)  # Convert all values to positive

                    # Replace NaN values with 0
                    depth_values = np.nan_to_num(depth_values, nan=0)
                print("depth map has been generated for mirrored image using the camera position used in depthmap of damaged half")

                # Save x, y, z coordinates
                # Load the CSV file containing red pixel coordinates (x, y)
                input_csv_path = "red_pixels_coordinates.csv"
                coordinates = []

                with open(input_csv_path, mode='r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    next(csv_reader)  # Skip header
                    for row in csv_reader:
                        x, y = map(int, row)  # Convert x and y to integers
                        coordinates.append((x, y))

                # Get the depth values (z-coordinates) from the depth map of mirrored_x.stl
                # Ensure depth_values have been computed from the mirrored_x depth map
                depth_values = np.nan_to_num(depth_values, nan=0)  # Fill NaNs with 0 if needed

                # Prepare output for x, y, z coordinates
                output_csv_path = "xyz_coordinates.csv"

                with open(output_csv_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(['x', 'y', 'z'])  # Write header

                    for x, y in coordinates:
                        # Ensure coordinates are within bounds
                        if 0 <= y < depth_values.shape[0] and 0 <= x < depth_values.shape[1]:
                            z = depth_values[y, x]  # Retrieve the depth value (z) at (x, y)
                            csv_writer.writerow([x, y, z])  # Write x, y, z to the CSV file
                        else:
                            print(f"Warning: Coordinate ({x}, {y}) is out of bounds and will be skipped.")

                print(f"Coordinates with z values saved to {output_csv_path}")
                progress.setValue(70)
                QApplication.processEvents()

                # Now use the CSV coordinates to create a 3D point cloud
                try:
                    # Read the CSV file with coordinates
                    csv_file_path = "xyz_coordinates.csv"
                    data = pd.read_csv(csv_file_path)

                    # Extract x, y, z coordinates
                    x_values, y_values, z_values = data['x'].values, data['y'].values, data['z'].values

                    # Create a PyVista point cloud
                    #points_cloud = pv.PolyData(list(zip(x_values, y_values, z_values)))
                    unique_points = np.unique(list(zip(x_values, y_values, z_values)), axis=0)
                    points_cloud = pv.PolyData(unique_points)

                    progress.setValue(80)
                    QApplication.processEvents()

                    # Convert point cloud to a surface mesh using Delaunay triangulation
                    try:
                        surface_mesh = points_cloud.delaunay_3d(alpha=2.0)
                        surface_mesh = surface_mesh.extract_surface()
                        # Apply thickness
                        solid_mesh = surface_mesh.extrude([0, 0, thickness], capping=True)

                        progress.setValue(90)
                        QApplication.processEvents()

                        # Save the implant mesh
                        implant_path = os.path.join(temp_dir, "implant.stl")
                        solid_mesh.save(implant_path)
                        print("3d model of peek has been generated")
                        progress.setValue(100)
                        QApplication.processEvents()

                        # Remove the progress bar
                        self.layout.removeWidget(progress)
                        progress.deleteLater()

                        # Open the implant view
                        self.show_implant(implant_path, thickness)

                    except Exception as e:
                        # Remove the progress bar
                        self.layout.removeWidget(progress)

                        progress.deleteLater()

                        self.show_error_message(f"Error generating implant: {str(e)}")
                        print(f"error:{str(e)}")

                    except Exception as e:
                        print(f"error:{str(e)}")

                except Exception as e:
                    print(f"error:{str(e)}")

        except Exception as e:
            print(f"error:{str(e)}")


    def show_implant(self, implant_path, thickness):
        """Open the implant view window"""
        try:
            self.implant_view = ImplantView(implant_path, self.damaged_path, self.undamaged_path,
                                            thickness, self.camera_info, self.algorithm)
            #self.implant_view.show()
            # Check if the current window is maximized
            if self.isMaximized():
                self.implant_view.showMaximized()
            else:
                self.implant_view.show()
            self.close()
        except Exception as e:
            self.show_error_message(f"Error showing implant: {str(e)}")
            print(f"error:{str(e)}")


    def go_back(self):
        self.damaged_model_view = DamagedModelView(self.damaged_path, self.undamaged_path,
                                                   self.camera_info, self.undamaged_side,
                                                   self.algorithm)
        #self.damaged_model_view.show()
        # Check if the current window is maximized
        if self.isMaximized():
            self.damaged_model_view.showMaximized()
        else:
            self.damaged_model_view.show()
        self.close()

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_()

class ImplantView(QMainWindow):
    def __init__(self, implant_path, damaged_path, undamaged_path, thickness, camera_info, algorithm):
        super().__init__()
        self.implant_path = implant_path
        self.damaged_path = damaged_path
        self.undamaged_path = undamaged_path
        self.thickness = thickness
        self.camera_info = camera_info
        self.algorithm = algorithm

        self.setWindowTitle("Generated Implant")
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet(AppStyle.MAIN_WINDOW_STYLE)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Header container
        # Header container
        header_container = QWidget()
        header_container = QWidget()
        header_container.setStyleSheet(f"""
                            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                                       stop:0 {AppStyle.PRIMARY_COLOR}, 
                                                       stop:1 #0077e6);
                            border-radius: {AppStyle.BORDER_RADIUS};
                            box-shadow: 0px 4px 10px rgba(0, 102, 204, 0.2);
                        """)

        # Header layout
        header_layout = QHBoxLayout(header_container)

        # Title Label
        title_label = QLabel("Generated Cranial Implant")
        title_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignLeft)
        header_layout.addWidget(title_label)

        # Icon Container
        icon_container = QWidget()
        icon_layout = QHBoxLayout(icon_container)

        # Icon Label
        icon_container = QWidget()
        icon_container.setFixedSize(50, 50)
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_label = QLabel("✅")
        icon_label.setStyleSheet(
            "color: white; font-size: 30px; background-color: rgba(255, 255, 255, 0.15); border-radius: 25px;")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(icon_label)
        # Add header to main layout
        main_layout.addWidget(header_container)

        header_layout.addWidget(icon_container)

        # Add header to main layout
        main_layout.addWidget(header_container)

        # Info
        info_label = QLabel(f"Implant generated using {algorithm} algorithm with thickness: {thickness:.2f} mm")
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)

        # Viewers layout
        viewers_layout = QHBoxLayout()

        # Damaged half viewer
        damaged_group = QGroupBox("Damaged Half")
        damaged_layout = QVBoxLayout()
        self.damaged_render_window = QVTKRenderWindowInteractor()
        damaged_layout.addWidget(self.damaged_render_window)
        damaged_group.setLayout(damaged_layout)
        viewers_layout.addWidget(damaged_group)

        # Implant viewer
        implant_group = QGroupBox("Generated Implant")
        implant_layout = QVBoxLayout()
        self.implant_render_window = QVTKRenderWindowInteractor()
        implant_layout.addWidget(self.implant_render_window)
        implant_group.setLayout(implant_layout)
        viewers_layout.addWidget(implant_group)

        main_layout.addLayout(viewers_layout)

        # Buttons layout
        buttons_layout = QHBoxLayout()

        # Back button
        back_button = QPushButton("← Exit To Homepage")
        back_button.setStyleSheet(AppStyle.BACK_BUTTON_STYLE)
        back_button.clicked.connect(self.go_back)
        buttons_layout.addWidget(back_button)

        buttons_layout.addStretch()

        # Save button
        save_button = QPushButton("Save Implant STL")
        save_button.setStyleSheet(AppStyle.PRIMARY_BUTTON_STYLE)
        save_button.clicked.connect(self.save_implant)
        buttons_layout.addWidget(save_button)

        # Generate Report button
        report_button = QPushButton("Generate Report")
        report_button.setStyleSheet(AppStyle.SECONDARY_BUTTON_STYLE)
        report_button.clicked.connect(self.generate_report)
        buttons_layout.addWidget(report_button)

        main_layout.addLayout(buttons_layout)

        # Container widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Setup renderers
        self.setup_damaged_renderer()
        self.setup_implant_renderer()

    def setup_damaged_renderer(self):
        try:
            # Create renderer for damaged half
            self.damaged_renderer = vtk.vtkRenderer()
            self.damaged_renderer.SetBackground(0.1, 0.1, 0.1)
            self.damaged_render_window.GetRenderWindow().AddRenderer(self.damaged_renderer)

            # Read the damaged half STL
            reader = vtk.vtkSTLReader()
            reader.SetFileName(self.damaged_path)
            reader.Update()

            if not reader.GetOutput():
                raise ValueError(f"Failed to read STL file: {self.damaged_path}")

            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.5, 0.5)  # Red color for damaged

            # Add actor to renderer
            self.damaged_renderer.AddActor(actor)

            # Set camera if provided
            if self.camera_info:
                camera = self.damaged_renderer.GetActiveCamera()
                camera.SetPosition(self.camera_info["position"])
                camera.SetFocalPoint(self.camera_info["focal_point"])
                camera.SetViewUp(self.camera_info["view_up"])
            else:
                self.damaged_renderer.ResetCamera()

            # Initialize interactor
            self.damaged_render_window.GetRenderWindow().Render()
            self.damaged_render_window.GetRenderWindow().GetInteractor().Initialize()

        except Exception as e:
            print(f"[Error] Failed to setup damaged renderer: {e}")


    def setup_implant_renderer(self):
        try:
            # Create renderer for implant
            self.implant_renderer = vtk.vtkRenderer()
            self.implant_renderer.SetBackground(0.1, 0.1, 0.1)
            self.implant_render_window.GetRenderWindow().AddRenderer(self.implant_renderer)

            # Read the implant STL
            reader = vtk.vtkSTLReader()
            reader.SetFileName(self.implant_path)
            reader.Update()

            if not reader.GetOutput():
                raise ValueError(f"Failed to read STL file: {self.implant_path}")

            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.5, 0.5)  # Blue color for implant

            # Add actor to renderer
            self.implant_renderer.AddActor(actor)

            self.implant_renderer.ResetCamera()

            # Initialize interactor
            self.implant_render_window.GetRenderWindow().Render()
            self.implant_render_window.GetRenderWindow().GetInteractor().Initialize()


        except Exception as e:
            print(f"[Error] Failed to setup implant renderer: {e}")



    def save_implant(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Implant STL", "", "STL Files (*.stl)")
        if file_name:
            try:
                # Copy the implant file to the selected location
                with open(self.implant_path, 'rb') as src_file, open(file_name, 'wb') as dst_file:
                    dst_file.write(src_file.read())

                QMessageBox.information(self, "Success", f"Implant saved to {file_name}")
                print("model has been saved successfully")
            except Exception as e:
                self.show_error_message(f"Error saving implant: {str(e)}")
                print(f"error:{str(e)}")
    homepage=0
    def go_back(self):
        self.homepage=1
        #self.main_window = SkullReconstruction()  # Open the previous window

        # Check if the current window is maximized
        #if self.isMaximized():
         #   self.main_window.showMaximized()
        #else:
         #   self.main_window.show()

        self.close()  # Close the current window

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_()

    def generate_report(self):
        """Generate a report with implant details - thickness, area, and curvature"""
        try:
            # Load the implant mesh to calculate area and curvature
            implant_mesh = pv.read(self.implant_path)

            # Calculate surface area (in mm²)
            area = implant_mesh.area

            # Get implant curvature (mean curvature)
            curvature = implant_mesh.curvature(curv_type='mean')
            mean_curvature = np.mean(curvature)

            # Create and style the report dialog
            report_dialog = QDialog(self)
            report_dialog.setWindowTitle("Implant Report")
            report_dialog.setMinimumWidth(500)
            report_dialog.setStyleSheet(f"""
                QDialog {{
                    background-color: {AppStyle.BACKGROUND_COLOR};
                    border-radius: {AppStyle.BORDER_RADIUS};
                }}
                QLabel {{
                    color: {AppStyle.TEXT_COLOR};
                    padding: 5px;
                }}
                QGroupBox {{
                    border: 1px solid {AppStyle.BORDER_COLOR};
                    border-radius: {AppStyle.BORDER_RADIUS};
                    margin-top: 10px;
                    background-color: {AppStyle.CARD_COLOR};
                }}
                QGroupBox::title {{
                    color: {AppStyle.PRIMARY_COLOR};
                    padding: 0 10px;
                }}
            """)

            # Create layout
            layout = QVBoxLayout(report_dialog)
            layout.setSpacing(15)

            # Header
            header = QLabel("IMPLANT REPORT")
            header.setStyleSheet(f"""
                font-size: 20px;
                font-weight: bold;
                color: {AppStyle.PRIMARY_COLOR};
                padding: 10px;
                border-bottom: 2px solid {AppStyle.BORDER_COLOR};
            """)
            header.setAlignment(Qt.AlignCenter)
            layout.addWidget(header)

            # Information groups
            info_group = QGroupBox()
            info_layout = QVBoxLayout()

            # Add details with consistent styling
            details = [
                ("Algorithm", self.algorithm),
                ("Thickness", f"{self.thickness:.2f} mm"),
                ("Surface Area", f"{area:.2f} mm")
            ]


            for label, value in details:
                detail_widget = QWidget()
                detail_layout = QHBoxLayout(detail_widget)

                label_widget = QLabel(f"{label}:")
                label_widget.setStyleSheet(f"color: {AppStyle.TEXT_COLOR}; font-weight: bold;")
                value_widget = QLabel(str(value))
                value_widget.setStyleSheet(f"""
                    color: {AppStyle.PRIMARY_COLOR};
                    background-color: rgba(26, 110, 156, 0.1);
                    border-radius: 4px;
                    padding: 5px 10px;
                """)

                detail_layout.addWidget(label_widget)
                detail_layout.addWidget(value_widget)
                detail_layout.addStretch()

                info_layout.addWidget(detail_widget)

            info_group.setLayout(info_layout)
            layout.addWidget(info_group)

            # Buttons
            button_layout = QHBoxLayout()

            save_button = QPushButton("Save Report")
            save_button.setStyleSheet(AppStyle.PRIMARY_BUTTON_STYLE)
            save_button.clicked.connect(lambda: self.save_report_from_dialog(report_dialog, details))

            close_button = QPushButton("Close")
            close_button.setStyleSheet(AppStyle.SECONDARY_BUTTON_STYLE)
            close_button.clicked.connect(report_dialog.close)

            button_layout.addWidget(save_button)
            button_layout.addWidget(close_button)
            layout.addLayout(button_layout)

            report_dialog.exec_()

        except Exception as e:
            self.show_error_message(f"Error generating report: {str(e)}")
            print(f"error:{str(e)}")


    def save_report_from_dialog(self, dialog, details):
        """Save the report from dialog to a text file"""
        file_name, _ = QFileDialog.getSaveFileName(dialog, "Save Report", "", "Text Files (*.txt)")
        if file_name:
            try:
                with open(file_name, 'w') as file:
                    file.write("IMPLANT REPORT\n")
                    file.write("-" * 30 + "\n")
                    for label, value in details:
                        file.write(f"{label}: {value}\n")
                    file.write("-" * 30 + "\n")
                QMessageBox.information(dialog, "Success", f"Report saved to {file_name}")
                print("report saved successfully")
            except Exception as e:
                self.show_error_message(f"Error saving report: {str(e)}")
                print(f"error:{str(e)}")

    def save_report(self, report_text):
        """Save the report to a text file"""
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "Text Files (*.txt)")
        if file_name:
            try:
                with open(file_name, 'w') as file:
                    file.write(report_text)
                QMessageBox.information(self, "Success", f"Report saved to {file_name}")
                #print("report has been saved")
            except Exception as e:
                self.show_error_message(f"Error saving report: {str(e)}")
                #print("error occured")

    def closeEvent(self, event):
        # Optional: Show confirmation
        if self.homepage==1:
            reply = QMessageBox.question(
                self,
                "Exit",
                "Are you sure you want to proceed to homepage?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.main_window = SkullReconstruction()  # Open the previous window

                # Check if the current window is maximized
                if self.isMaximized():
                    self.main_window.showMaximized()
                else:
                    self.main_window.show()
            else:
                event.ignore()
                self.homepage=0
        else:
            reply = QMessageBox.question(
                self,
                "Exit",
                "Are you sure you want to close the application?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                try:
                    if self.damaged_render_window:
                        self.damaged_render_window.GetRenderWindow().Finalize()
                        self.damaged_render_window = None
                    if self.implant_render_window:
                        self.implant_render_window.GetRenderWindow().Finalize()
                        self.implant_render_window = None
                except Exception as e:
                    print(f"[Warning] Renderer cleanup failed: {e}")
                event.accept()
            else:
                event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    skull_reconstruction = SkullReconstruction()
    skull_reconstruction.show()
    sys.exit(app.exec_())
