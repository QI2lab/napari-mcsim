"""
GUI code for napari plugin for Structured illumination microscopy (SIM) reconstruction using mcsim
"""

from qtpy import QtWidgets as QtW
from qtpy.QtWidgets import QWidget
from qtpy import uic, QtCore
import napari.viewer

from pathlib import Path
import tifffile
import zarr
from io import StringIO
import numpy as np
import threading
from mcsim.analysis.sim_reconstruction import SimImageSet
import localize_psf.fit_psf as fpsf


# wrap printing to textEdit wdiget to look like file.
class TextEditStringIO(StringIO):
    def __init__(self, textEdit: QtW.QTextEdit):
        super().__init__()
        self.textEdit = textEdit

    def getvalue(self) -> str:
        return self.textEdit.toPlainText()

    def write(self, __s: str) -> int:
        # self.textEdit.append(__s)
        self.textEdit.insertPlainText(__s)


class MCSIM(QWidget):
    _ui_file = str(Path(__file__).parent / "mcsim_plugin.ui")

    def __init__(self, viewer: napari.viewer.Viewer, parent=None):
        super().__init__(parent=parent)
        # load GUI elements from file. Can also add these as class elements to add with linting and etc.
        uic.loadUi(self._ui_file, self)
        self.viewer = viewer

        # SIM instance
        self.sim = None
        self.fx = None
        self.fy = None

        # layers
        self.raw_layer = None
        self.fft_layer = None
        self.widefield_layer = None
        self.decon_layer = None
        self.mcnr_layer = None
        self.sim_os_layer = None
        self.sim_sr_layer = None

        # threading
        self.sim_lock = threading.Lock() # must acquire this lock before modifying self.sim
        self.worker_thread = None

        # print results
        self.textEdit.setReadOnly(True)
        self.stream = TextEditStringIO(self.textEdit) # print("test stream", file=self.stream)

        # disconnect napari update coordinate display and replace with our own
        self.viewer.cursor.events.position.disconnect()
        self.viewer.cursor.events.position.connect(self._update_status_bar_from_cursor)

        # saving and loading data
        self.supported_file_types = ["tiff", "zarr", "hd5f"]
        self.file_type_comboBox.addItems(self.supported_file_types)
        self.file_type_comboBox.setCurrentText(self.supported_file_types[0])
        self.browse_Button.clicked.connect(self._browse_img_files)
        # update layer list when layers change
        self.viewer.events.layers_change.connect(self._refresh_img_layers) # todo: this signal is deprecated
        # self.viewer.layers.events.changed.connect(self._refresh_layers)
        self.initialize_Button.clicked.connect(self._initialize_sim)
        self.browse_LineEdit.textChanged.connect(self._refresh_img_arrays)
        # todo: detect number of dimensions and set those combo boxes
        self.browse_LineEdit.textChanged.connect(self._refresh_axis_detection)
        self.array_select_comboBox.currentIndexChanged.connect(self._refresh_axis_detection)
        self.layer_comboBox.currentIndexChanged.connect(self._refresh_axis_detection)
        # dimension definition
        self.dimension_types = [
                                "other",
                                "angles",
                                "phases",
                                "channels",
                                "phases,angles",
                                "angles,phases",
                                "phases,angles,channels",
                                "angles,phases,channels",
                                "channels,phases,angles",
                                "channels,angles,phases"
                                ]

        # necessary settings
        self.pixel_size_doubleSpinBox.setValue(0.065)
        self.wavelength_doubleSpinBox.setValue(532.)
        self.na_doubleSpinBox.setValue(1.3)
        self.nangles_spinBox.setValue(3)
        self.nphases_spinBox.setValue(3)

        # PSF/OTF
        self.browse_otf_pushButton.clicked.connect(self._browse_psf_otf)
        self.otf_format_comboBox.addItems(self.supported_file_types)
        self.psf_mode_comboBox.addItems(["PSF", "OTF"])
        self.viewer.events.layers_change.connect(self._refresh_otf_layers)
        self.otf_lineEdit.textChanged.connect(self._refresh_otf_arrays)
        self.set_otf_pushButton.clicked.connect(self._load_otf)

        # SIM parameters
        self.select_pushButton.clicked.connect(self._select_parameters_mouse)
        self.add_pushButton.clicked.connect(self._add_parameters)
        self.remove_pushButton.clicked.connect(self._remove_parameters)
        self.clear_pushButton.clicked.connect(self._clear_parameters)

        # reconstruct
        self.phase_comboBox.addItems(SimImageSet.allowed_phase_estimation_modes)
        self.frq_comboBox.addItems(SimImageSet.allowed_frq_estimation_modes)
        self.recon_mode_comboBox.addItems(SimImageSet.allowed_reconstruction_modes)
        self.band_replacement_doubleSpinBox.setValue(0.4)
        self.wiener_doubleSpinBox.setValue(0.1)

        self.reconstruct_pushButton.clicked.connect(self._reconstruct_sim)
        self.show_recon_pushButton.clicked.connect(self._show)

        # saving
        self.select_save_dir_pushButton.clicked.connect(self._browse_save_dir)
        self.save_format_comboBox.addItems(self.supported_file_types)

    def _convert_pixel_to_frq(self, cy, cx):

        if self.fx is None:
            fx = np.nan
        else:
            fxs = self.fx
            ix_low = int(np.floor(cx))
            ix_high = int(np.ceil(cx))

            try:
                fx = fxs[ix_low] * (ix_high - cx) + fxs[ix_high] * (cx - ix_low)
            except IndexError:
                fx = np.nan

        if self.fy is None:
            fy = np.nan
        else:
            fys = self.fy
            iy_low = int(np.floor(cy))
            iy_high = int(np.ceil(cy))
            try:
                fy = fys[iy_low] * (iy_high - cy) + fys[iy_high] * (cy - iy_low)
            except IndexError:
                fy = np.nan

        return fy, fx

    def _update_status_bar_from_cursor(self, event):
        """
        Replacement for viewer function which knows about the FFT coordinates
        :param event:
        :return:
        """
        # Update status and help bar based on active layer

        if not self.viewer.mouse_over_canvas:
            return

        active = self.viewer.layers.selection.active
        fft_layer_active = active is not None and active == self.fft_layer and self.sim is not None

        if active is not None:
            # call the usual function
            # todo: this throws warning that this attribute will not be accessible after version 0.5.0
            # self.viewer._update_status_bar_from_cursor(event)

            cursor_pos = self.viewer.cursor.position # canvas coords
            cursor_pos_data = active.world_to_data(cursor_pos)

            st = active.get_status(
                cursor_pos,
                view_direction=None, #self.viewer.cursor._view_direction,
                dims_displayed=list(self.viewer.dims.displayed),
                world=True,
            )

            if fft_layer_active:
                # convert to FFT frequency
                cy, cx = cursor_pos_data[-2:]
                fy, fx, = self._convert_pixel_to_frq(cy, cx)
                st["coordinates"] = f"(fy, fx) = [{fy:.3f} {fx:.3f}]" + st["coordinates"]

            self.viewer.status = st

            # todo: update this part too...
            self.viewer.help = active.help
            if self.viewer.tooltip.visible:
                self.viewer.tooltip.text = active._get_tooltip_text(
                    self.viewer.cursor.position,
                    view_direction=None, #self.viewer.cursor._view_direction,
                    dims_displayed=list(self.dims.displayed),
                    world=True,
                )

    def browse_files(self, file_type):

        if file_type == "tiff":
            fname = str(QtW.QFileDialog.getOpenFileName(self, "", "", "tiff images (*.tiff *.tif)")[0])
        elif file_type == "zarr":
            fname = str(QtW.QFileDialog.getExistingDirectory(self, "", "zarr file"))
        elif file_type == "hd5f":
            fname = str(QtW.QFileDialog.getOpenFileName(self, "", "", "hd5f (*.hd5f)")[0])
        else:
            raise ValueError()

        return fname

    def _browse_img_files(self):
        fname = self.browse_files(self.file_type_comboBox.currentText())
        self.browse_LineEdit.setText(fname)

    def _refresh_img_layers(self):
        self.layer_comboBox.clear()
        layer_names = [""] + [l.name for l in self.viewer.layers]
        self.layer_comboBox.addItems(layer_names)

    def _get_arrays(self, fname):
        if fname.suffix == ".tiff" or fname.suffix == ".tif":
            return None

        if fname.suffix == ".zarr":
            z = zarr.open(fname, "r")
            array_names = [arr.name for n, arr in z.arrays(recurse=True)]

        elif fname.suffix == ".h5py":
            array_names = None
            raise NotImplementedError()
        else:
            raise ValueError(f"file type {fname.suffix:s} not supported")

        return array_names

    def _load_file(self, fname, array_name=None):

        if fname.suffix == ".tiff" or fname.suffix == ".tif":
            try:
                imgs = tifffile.imread(fname)
            except ValueError as e:
                print(e, file=self.stream)
                print(e)
                return

        elif fname.suffix == ".zarr":
            try:
                z = zarr.open(fname, "r")
            except zarr.errors.PathNotFoundError as e:
                print(e, file=self.stream)
                print(e)
                return

            # get array
            path_items = [s for s in array_name.split("/") if s]
            imgs = z
            for p in path_items:
                imgs = imgs[p]

        elif fname.suffix == ".hd5f":
            # z = h5py.
            raise NotImplementedError()
        else:
            raise ValueError(f"file_type={fname.suffix:s} is not supported")

        return imgs

    def _get_data_shape(self, fname, array_name=None):

        if fname.suffix == ".tiff" or fname.suffix == ".tif":
            try:
                # todo: don't actually load images!
                imgs = tifffile.imread(fname)
                shape = imgs.shape
            except ValueError as e:
                print(e)
                return

        elif fname.suffix == ".zarr":
            try:
                z = zarr.open(fname, "r")
            except zarr.errors.PathNotFoundError as e:
                print(e)
                return

            # get array
            path_items = [s for s in array_name.split("/") if s]
            imgs = z
            for p in path_items:
                imgs = imgs[p]
            shape = imgs.shape

        elif fname.suffix == ".hd5f":
            # z = h5py.
            raise NotImplementedError()
        else:
            raise ValueError(f"file_type={fname.suffix:s} is not supported")

        return shape

    def _refresh_img_arrays(self):
        self.array_select_comboBox.clear()
        fname = Path(self.browse_LineEdit.text())
        array_names = self._get_arrays(fname)

        if array_names is not None:
            self.array_select_comboBox.addItems(array_names)

    def _refresh_axis_detection(self):

        # find selection
        # if layer is selected, load this. Otherwise try to load file
        layer_selected = self.layer_comboBox.currentText()
        layer_list = [l for l in self.viewer.layers if l.name == layer_selected]

        if layer_list == []:
            layer = None
        else:
            layer = layer_list[0]

        if layer is not None:
            ndim = layer.data.ndim
            shape = layer.data.shape
        else:
            fname = Path(self.browse_LineEdit.text())
            array_name = self.array_select_comboBox.currentText()

            if not fname:
                return

            shape = self._get_data_shape(fname, array_name)
            ndim = len(shape)

        self.dimension_tableWidget.clearContents()
        self.dimension_tableWidget.setRowCount(0)

        for ii in range(ndim):
            self.dimension_tableWidget.insertRow(ii)

            # size
            size_label = QtW.QLabel(self)
            size_label.setText(str(shape[ii]))
            self.dimension_tableWidget.setCellWidget(ii, 0, size_label)

            # selector box
            selector_comboBox = QtW.QComboBox(self)
            selector_comboBox.addItems(self.dimension_types)
            self.dimension_tableWidget.setCellWidget(ii, 1, selector_comboBox)

            # name box
            name_lineEdit = QtW.QLineEdit(self)
            self.dimension_tableWidget.setCellWidget(ii, 2, name_lineEdit)

            # usual angle/phase dims
            if ii == ndim - 3 and ndim == 3:
                selector_comboBox.setCurrentText("phases,angles")
            elif ii == ndim - 3 and ndim > 3:
                selector_comboBox.setCurrentText("phases")

            if ii == ndim - 4:
                selector_comboBox.setCurrentText("angles")

            # x and y dimensions
            if ii == ndim - 2:
                selector_comboBox.setCurrentText("other")
                name_lineEdit.setText("y")
            elif ii == ndim - 1:
                selector_comboBox.setCurrentText("other")
                name_lineEdit.setText("x")

        # self.axes_comboBox.clear()
        # self.channel_axis_comboBox.clear()
        #
        # dims = [""] + [str(a) for a in list(range(ndim))]
        # self.axes_comboBox.addItems(dims)
        # self.channel_axis_comboBox.addItems(dims)

    def _detect_dimensions(self):
        """
        Return information about how to reshape and change dimensions of dataset. This because loaded data may combine
        several different attributes along one axis (e.g. angle and phase and channel). The data should first be
        reshaped using the output_shape, and then axes transposed using the transpose_axes.

        :return:
        output_shape
        transpose_axes
        output_axes_names
        """
        nrows = self.dimension_tableWidget.rowCount()

        nangles = self.nangles_spinBox.value()
        nphases = self.nphases_spinBox.value()
        nsim = nangles * nphases

        # original
        sizes = []
        mode = []
        description = []
        for ii in range(nrows):
            sizes.append(int(self.dimension_tableWidget.cellWidget(ii, 0).text()))
            mode.append(self.dimension_tableWidget.cellWidget(ii, 1).currentText())
            description.append(self.dimension_tableWidget.cellWidget(ii, 2).text())

        # todo: less brute force ways to do this ... but this is fine for now
        # insert dimensions in opposite order, because strings are ordered from fastest-to-slowest-axis
        # while python array ordered from slowest-to-fastest axis
        output_shape = []
        output_axes_names = []
        for ii in range(nrows):
            if mode[ii] == "angles":
                output_shape.append(nangles)
                output_axes_names.append("angles")
            elif mode[ii] == "phases":
                output_shape.append(nphases)
                output_axes_names.append("phases")
            elif mode[ii] == "angles,phases":
                output_shape.append(nphases)
                output_shape.append(nangles)
                output_axes_names.append("phases")
                output_axes_names.append("angles")
            elif mode[ii] == "phases,angles":
                output_shape.append(nphases)
                output_shape.append(nangles)
                output_axes_names.append("angles")
                output_axes_names.append("phases")
            elif mode[ii] == "angles,phases,channels":
                output_shape.append(sizes[ii] // nsim)
                output_shape.append(nphases)
                output_shape.append(nangles)
                output_axes_names.append("channels")
                output_axes_names.append("phases")
                output_axes_names.append("angles")
            elif mode[ii] == "phases,angles,channels":
                output_shape.append(sizes[ii] // nsim)
                output_shape.append(nangles)
                output_shape.append(nphases)
                output_axes_names.append("channels")
                output_axes_names.append("angles")
                output_axes_names.append("phases")
            elif mode[ii] == "channels,phases,angles":
                output_shape.append(nangles)
                output_shape.append(nphases)
                output_shape.append(sizes[ii] // nsim)
                output_axes_names.append("angles")
                output_axes_names.append("phases")
                output_axes_names.append("channels")
            elif mode[ii] == "channels,angles,phases":
                output_shape.append(nphases)
                output_shape.append(nangles)
                output_shape.append(sizes[ii] // nsim)
                output_axes_names.append("phases")
                output_axes_names.append("angles")
                output_axes_names.append("channels")
            else:
                output_shape.append(sizes[ii])
                output_axes_names.append(description[ii])

        transpose_axes = np.arange(len(output_axes_names))
        n_final_axes = len(transpose_axes)
        desired_phase_axis = n_final_axes - 3
        desired_angles_axis = n_final_axes - 4

        # get current phase axis
        phases_axis = [ii for ii, p in enumerate(output_axes_names) if p == "phases"]
        if len(phases_axis) > 1:
            raise ValueError("multiple instances of 'phases' axis detected")
        elif len(phases_axis) == 0:
            raise ValueError("no 'phases' axis detected")
        phases_axis = phases_axis[0]

        # swap if necessary
        if phases_axis != desired_phase_axis:
            transpose_axes[phases_axis] = desired_phase_axis
            transpose_axes[desired_phase_axis] = phases_axis
            # swap descriptions
            output_axes_names[phases_axis], output_axes_names[desired_phase_axis] = output_axes_names[desired_phase_axis], output_axes_names[phases_axis]

        # get current angle axis (after phase axis correction this may be different)
        angles_axis = [ii for ii, p in enumerate(output_axes_names) if p == "angles"]
        if len(angles_axis) > 1:
            raise ValueError("multiple instances of 'angles' axis detected")
        elif len(angles_axis) == 0:
            raise ValueError("no 'angles' axis detected")
        angles_axis = angles_axis[0]

        # swap
        if angles_axis != n_final_axes - 4:
            transpose_axes[angles_axis] = desired_angles_axis
            transpose_axes[desired_angles_axis] = angles_axis
            output_axes_names[angles_axis], output_axes_names[desired_angles_axis] = output_axes_names[desired_angles_axis], output_axes_names[angles_axis]

        return output_shape, transpose_axes, output_axes_names

    def _initialize_sim(self):
        # if layer is selected, load this. Otherwise try to load file
        layer_selected = self.layer_comboBox.currentText()
        layer_list = [l for l in self.viewer.layers if l.name == layer_selected]

        if layer_list == []:
            layer = None
        else:
            layer = layer_list[0]

        if layer is not None:
            imgs = layer.data
        else:
            fname = Path(self.browse_LineEdit.text())
            array_name = self.array_select_comboBox.currentText()

            if not fname:
                return

            imgs = self._load_file(fname, array_name)

        # get axes and reshape to correct size
        reshape_size, transpose_axes, axes_names = self._detect_dimensions()
        try:
            imgs = imgs.reshape(reshape_size).transpose(transpose_axes)
        except ValueError as e:
            print(e)
            print("check dimension order definition for errors")
            print(e, file=self.stream)
            print("check dimension order definition for errors", file=self.stream)
            return

        def _preprocess():
            with self.sim_lock:
                self.sim = SimImageSet(use_gpu=self.gpu_checkBox.isChecked(),
                                       print_to_terminal=True)
                self.sim.add_stream(self.stream) # print to widget
                self.sim.preprocess_data(pix_size_um=self.pixel_size_doubleSpinBox.value(),
                                         na=self.na_doubleSpinBox.value(),
                                         wavelength=self.wavelength_doubleSpinBox.value() / 1e3, # in um
                                         imgs=imgs,
                                         normalize_histograms=True,
                                         gain=2,
                                         background=100,
                                         axes_names=axes_names)

                # store frequency elsewhere so can access without acquiring lock
                self.fx = self.sim.fx
                self.fy = self.sim.fy

        # main advantage of threading here is GUI will update immediately with printed results
        if self.worker_thread is not None:
            self.worker_thread.join()

        self.worker_thread = threading.Thread(target=_preprocess)
        self.worker_thread.start()

        # todo: want to send signal when thread finishes and then show, but this will do for now
        self.worker_thread.join()
        self._show()

    def _refresh_otf_layers(self):
        self.otf_layer_comboBox.clear()
        layer_names = [""] + [l.name for l in self.viewer.layers]
        self.otf_layer_comboBox.addItems(layer_names)

    def _refresh_otf_arrays(self):
        self.otf_array_comboBox.clear()
        fname = Path(self.otf_lineEdit.text())
        array_names = self._get_arrays(fname)

        if array_names is not None:
            self.otf_array_comboBox.addItems(array_names)

    def _browse_psf_otf(self):
        file_type = self.otf_format_comboBox.currentText()
        fname = self.browse_files(file_type)
        self.otf_lineEdit.setText(fname)

    def _load_otf(self):
        # get possible sources of OTF/PSF
        layer_selected = self.otf_layer_comboBox.currentText()
        layer_list = [l for l in self.viewer.layers if l.name == layer_selected]

        if layer_list == []:
            layer = None
        else:
            layer = layer_list[0]

        fname = Path(str(self.otf_lineEdit.text()))
        array_name = str(self.otf_array_comboBox.currentText())

        generate_from_params = layer is not None and not fname
        if generate_from_params:
            otf = None
        else:
            if layer is not None:
                data = layer.data
            elif fname:
                data = self._load_file(fname, array_name)

            mode = self.psf_mode_comboBox.currentText()
            if mode == "PSF":
                psf = data
                otf, _ = fpsf.psf2otf(psf)
            elif mode == "OTF":
                otf = data
            else:
                raise ValueError(f"mode={mode:s} is not supported")

        with self.sim_lock:
            self.sim.update_otf(otf)

    def _refresh_channel_axes(self):
        pass

    # add, remove, clear, move_to parameters table
    def _select_parameters_mouse(self):
        # self.viewer.mouse_drag_callbacks.append(self._add_parameter_by_point)
        self.viewer.mouse_double_click_callbacks.append(self._add_parameter_by_point)

        try:
            self.select_pushButton.clicked.disconnect()
        except Exception as e:
            print(e)

        self.select_pushButton.clicked.connect(self._stop_select_parameters_mouse)
        self.select_pushButton.setText("Stop selecting...")

    def _stop_select_parameters_mouse(self):
        # self.viewer.mouse_drag_callbacks.pop()
        self.viewer.mouse_double_click_callbacks.pop()

        try:
            self.select_pushButton.clicked.disconnect()
        except Exception as e:
            print(e)

        self.select_pushButton.clicked.connect(self._select_parameters_mouse)
        self.select_pushButton.setText("Select")

    def _add_parameter_by_point(self, viewer, event):
        # get canvas coordinate
        canvas_coord = event.position

        # convert to layer coordinate
        if self.fft_layer is not None:
            # layer_name = "test"
            # layer = [l for l in self.viewer.layers if l.name == layer_name][0]
            data_coord = self.fft_layer.world_to_data(canvas_coord)

            cy, cx = data_coord[-2:]
            fy, fx, = self._convert_pixel_to_frq(cy, cx)
            angle_ind = data_coord[-4]

            # grab peak phases as estimate
            with self.sim_lock:
                # leading coords
                slices = [slice(int(dc), int(dc + 1)) for ii, dc in enumerate(data_coord) if ii < len(data_coord) - 3]

                cy_round = int(np.round(cy))
                cx_round = int(np.round(cx))
                slices = slices + [slice(None)] + [slice(cy_round, cy_round + 1), slice(cx_round, cx_round + 1)]

                # todo: slow due to dask ... maybe should embed this info in layer instead
                phases_deg = np.mod(np.angle(self.sim.imgs_ft[tuple(slices)].compute().squeeze()), 2*np.pi) * 180/np.pi

            # add to table
            self._add_parameters()
            idx = self.parameter_tableWidget.rowCount() - 1
            self.parameter_tableWidget.cellWidget(idx, 1).setValue(fy)
            self.parameter_tableWidget.cellWidget(idx, 0).setValue(fx)
            self.parameter_tableWidget.cellWidget(idx, 6).setValue(angle_ind)
            self.parameter_tableWidget.cellWidget(idx, 2).setValue(phases_deg[0])
            self.parameter_tableWidget.cellWidget(idx, 3).setValue(phases_deg[1])
            self.parameter_tableWidget.cellWidget(idx, 4).setValue(phases_deg[2])

            # jump viewer to next angle
            ndim = self.viewer.dims.ndim
            new_step = list(self.viewer.dims.current_step)[ndim - 4] + 1
            # if this is beyond the axis, the viewer will stay where it is
            self.viewer.dims.set_current_step(axis=ndim - 4, value=new_step)

    def _add_parameters(self):
        idx = self.parameter_tableWidget.rowCount()
        self.parameter_tableWidget.insertRow(idx)

        # create a combo_box for channels in the table
        fx_doubleSpinBox = QtW.QDoubleSpinBox(self)
        fx_doubleSpinBox.setMinimum(-1e6)
        fx_doubleSpinBox.setMaximum(1e6)
        fx_doubleSpinBox.setSingleStep(0.01)
        fx_doubleSpinBox.setDecimals(3)

        fy_doubleSpinBox = QtW.QDoubleSpinBox(self)
        fy_doubleSpinBox.setMinimum(-1e6)
        fy_doubleSpinBox.setMaximum(1e6)
        fy_doubleSpinBox.setSingleStep(0.01)
        fy_doubleSpinBox.setDecimals(3)

        phase1_doubleSpinBox = QtW.QDoubleSpinBox(self)
        phase1_doubleSpinBox.setMinimum(-1e6)
        phase1_doubleSpinBox.setMaximum(1e6)
        phase1_doubleSpinBox.setSingleStep(10)
        phase1_doubleSpinBox.setDecimals(2)
        phase1_doubleSpinBox.setValue(0.)

        phase2_doubleSpinBox = QtW.QDoubleSpinBox(self)
        phase2_doubleSpinBox.setMinimum(-1e6)
        phase2_doubleSpinBox.setMaximum(1e6)
        phase2_doubleSpinBox.setSingleStep(10)
        phase2_doubleSpinBox.setDecimals(2)
        phase2_doubleSpinBox.setValue(120.)

        phase3_doubleSpinBox = QtW.QDoubleSpinBox(self)
        phase3_doubleSpinBox.setMinimum(-1e6)
        phase3_doubleSpinBox.setMaximum(1e6)
        phase3_doubleSpinBox.setSingleStep(10)
        phase3_doubleSpinBox.setDecimals(2)
        phase3_doubleSpinBox.setValue(240.)

        mod_depth_doubleSpinBox = QtW.QDoubleSpinBox(self)
        mod_depth_doubleSpinBox.setMinimum(0)
        mod_depth_doubleSpinBox.setMaximum(1)
        mod_depth_doubleSpinBox.setSingleStep(0.1)
        mod_depth_doubleSpinBox.setDecimals(3)
        mod_depth_doubleSpinBox.setValue(1.)

        angle_doubleSpinBox = QtW.QDoubleSpinBox(self)
        angle_doubleSpinBox.setMinimum(0)
        angle_doubleSpinBox.setMaximum(1e6)
        angle_doubleSpinBox.setSingleStep(1)
        angle_doubleSpinBox.setDecimals(0)
        angle_doubleSpinBox.setValue(idx)

        channel_doubleSpinBox = QtW.QDoubleSpinBox(self)
        channel_doubleSpinBox.setMinimum(0)
        channel_doubleSpinBox.setMaximum(1e6)
        channel_doubleSpinBox.setSingleStep(1)
        channel_doubleSpinBox.setDecimals(0)

        self.parameter_tableWidget.setCellWidget(idx, 0, fx_doubleSpinBox)
        self.parameter_tableWidget.setCellWidget(idx, 1, fy_doubleSpinBox)
        self.parameter_tableWidget.setCellWidget(idx, 2, phase1_doubleSpinBox)
        self.parameter_tableWidget.setCellWidget(idx, 3, phase2_doubleSpinBox)
        self.parameter_tableWidget.setCellWidget(idx, 4, phase3_doubleSpinBox)
        self.parameter_tableWidget.setCellWidget(idx, 5, mod_depth_doubleSpinBox)
        self.parameter_tableWidget.setCellWidget(idx, 6, angle_doubleSpinBox)
        self.parameter_tableWidget.setCellWidget(idx, 7, channel_doubleSpinBox)

    def _remove_parameters(self):
        # remove selected position
        rows = {r.row() for r in self.parameter_tableWidget.selectedIndexes()}
        for idx in sorted(rows, reverse=True):
            self.parameter_tableWidget.removeRow(idx)

    def _clear_parameters(self):
        # clear all positions
        self.parameter_tableWidget.clearContents()
        self.parameter_tableWidget.setRowCount(0)

    def _reconstruct_sim(self):
        # reconstruction settings
        compute_widefield = self.widefield_checkBox.isChecked()
        compute_os = self.simos_checkBox.isChecked()
        compute_deconvolve = self.deconvolve_checkBox.isChecked()
        compute_mcnr = self.mcnr_checkBox.isChecked()
        phase_mode = self.phase_comboBox.currentText()
        frq_mode = self.frq_comboBox.currentText()
        recon_mode = self.recon_mode_comboBox.currentText()
        wiener_param = self.wiener_doubleSpinBox.value()
        fmax_exclude_band0 = self.band_replacement_doubleSpinBox.value()

        # grab guess parameters
        nfrqs = self.parameter_tableWidget.rowCount()

        frq_guess = np.zeros((3, 2))
        phases_guess = np.zeros((3, 3))
        mod_depths_guess = np.zeros((3))
        for ii in range(nfrqs):
            frq_guess[ii, 0] = self.parameter_tableWidget.cellWidget(ii, 0).value()
            frq_guess[ii, 1] = self.parameter_tableWidget.cellWidget(ii, 1).value()
            phases_guess[ii, 0] = self.parameter_tableWidget.cellWidget(ii, 2).value()
            phases_guess[ii, 1] = self.parameter_tableWidget.cellWidget(ii, 3).value()
            phases_guess[ii, 2] = self.parameter_tableWidget.cellWidget(ii, 4).value()
            mod_depths_guess[ii] = self.parameter_tableWidget.cellWidget(ii, 5).value()
        # phases to radians
        phases_guess = phases_guess * np.pi / 180.

        def _recon_and_save():
            with self.sim_lock:
                if self.sim.otf is None:
                    self.sim.update_otf()

                # todo: add freedom to set more of these parameters
                self.sim.update_recon_settings(wiener_parameter=wiener_param,
                                               frq_estimation_mode=frq_mode,
                                               phase_estimation_mode=phase_mode,
                                               combine_bands_mode="fairSIM",
                                               fmax_exclude_band0=fmax_exclude_band0,
                                               use_fixed_mod_depths=False,
                                               minimum_mod_depth=0.3,
                                               determine_amplitudes=False,
                                               min_p2nr=1,
                                               trim_negative_values=True)

                self.sim.update_parameter_guesses(frq_guess=frq_guess,
                                                  phases_guess=phases_guess,
                                                  mod_depths_guess=mod_depths_guess)

                # todo: need way to set slices for e.g. z-stacks
                self.sim.reconstruct(slices=None,
                                     compute_widefield=compute_widefield,
                                     compute_os=compute_os,
                                     compute_deconvolved=compute_deconvolve,
                                     compute_mcnr=compute_mcnr)

                self.sim.print_parameters()

                # saving
                if self.save_groupBox.isChecked():
                    save_dir = Path(self.save_lineEdit.text())
                    prefix = self.save_prefix_lineEdit.text()
                    suffix = self.save_suffix_lineEdit.text()

                    print("saving reconstructed images...", file=self.stream)
                    print("saving reconstructed images...")
                    self.sim.save_imgs(save_dir=save_dir,
                                       save_suffix=suffix,
                                       save_prefix=prefix,
                                       format=self.save_format_comboBox.currentText(),
                                       save_patterns=False,
                                       save_raw_data=False,
                                       save_processed_data=False)

                    if self.save_diagnostic_plots_checkBox.isChecked():
                        print("saving diagnostic images...", file=self.stream)
                        print("saving diagnostic images...")
                        self.sim.plot_figs(save_dir,
                                           save_prefix=prefix,
                                           save_suffix=suffix,
                                           diagnostics_only=True,  # turn off saving SIM reconstruction figure
                                           interactive_plotting=False,
                                           figsize=(20, 10),
                                           imgs_dpi=300)

        # if thread is already running, let it finish
        if self.worker_thread is not None:
            self.worker_thread.join()

        self.worker_thread = threading.Thread(target=_recon_and_save,
                                              args=())
        self.worker_thread.start()

    def _show(self):

        with self.sim_lock:
            if self.sim is None:
                return

            if self.sim.imgs is not None:
                if self.raw_layer is None or self.raw_layer not in self.viewer.layers:
                    self.raw_layer = self.viewer.add_image(self.sim.imgs,
                                                           name="raw SIM data")
                else:
                    self.raw_layer.data = self.sim.imgs

                # set axes names
                try:
                    self.viewer.dims.axis_labels = self.sim.axes_names
                except Exception as e:
                    print(e, file=self.stream)
                    print(e)

                # set to first slices
                for id in range(self.sim.imgs.ndim - 2):
                    self.viewer.dims.set_current_step(axis=id, value=0)

            if self.sim.imgs_ft is not None:
                if self.fft_layer is None or self.fft_layer not in self.viewer.layers:
                    # todo: gamma will not update until slider is moved, see https://github.com/napari/napari/issues/1866
                    self.fft_layer = self.viewer.add_image(abs(self.sim.imgs_ft),
                                                           name="raw SIM data fft",
                                                           translate=[0, self.sim.nx],
                                                           gamma=0.2)
                else:
                    self.fft_layer.data = abs(self.sim.imgs_ft)

            if self.sim.widefield is not None:
                if self.widefield_layer is None or self.widefield_layer not in self.viewer.layers:
                    self.widefield_layer = self.viewer.add_image(self.sim.widefield,
                                                                 name="SIM widefield")
                else:
                    self.widefield_layer.data = self.sim.widefield

            if self.sim.widefield_deconvolution is not None:
                if self.decon_layer is None or self.decon_layer not in self.viewer.layers:
                    self.decon_layer = self.viewer.add_image(self.sim.widefield_deconvolution,
                                                             name="SIM deconvolved")
                else:
                    self.decon_layer.data = self.sim.widefield_deconvolution

            if self.sim.mcnr is not None:
                if self.mcnr_layer is None or self.mcnr_layer not in self.viewer.layers:
                    self.mcnr_layer = self.viewer.add_image(self.sim.mcnr,
                                                            name="MCNR")
                else:
                    self.mcnr_layer.data = self.sim.mcnr

            if self.sim.sim_os is not None:
                if self.sim_os_layer is None or self.sim_os_layer not in self.viewer.layers:
                    self.sim_os_layer = self.viewer.add_image(self.sim.sim_os,
                                                              name="SIM-OS")
                else:
                    self.sim_os_layer.data = self.sim.sim_os

            if self.sim.sim_sr is not None:
                if self.sim_sr_layer is None or self.sim_sr_layer not in self.viewer.layers:
                    self.sim_sr_layer = self.viewer.add_image(self.sim.sim_sr,
                                                              scale=(0.5, 0.5),
                                                              translate=(0, 0),  # todo: correct
                                                              name="SIM-SR")
                else:
                    self.sim_sr_layer.data = self.sim.sim_sr

    def _browse_save_dir(self):
        file_dir = str(QtW.QFileDialog.getExistingDirectory(self, "", "save directory"))
        self.save_lineEdit.setText(file_dir)
