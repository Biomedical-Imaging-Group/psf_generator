# Imaging a Dipole: the Detection Path

The propagators describe the *focusing* path of a microscope: a pupil function $\mathbf{e}_\infty(\mathbf{s})$ is turned into the field $\mathbf{E}(\boldsymbol{\rho})$ around the focus by the Richards-Wolf integral [%s](#eq:initial-vectorial).
Many applications need the reverse, *imaging* path: a dipole radiating in the sample (a fluorophore, or the dipole induced in a nanoparticle by the illumination) is collected by the objective and imaged onto a camera by the tube lens.
Both paths are governed by the same Debye-Wolf integral, but running it backwards changes a few geometric factors, which we derive here following {cite:p}`foreman2011computational, torok1998theory, Novotny_Hecht_2012`.
The dipole imagers of `psf_generator.imaging` implement this path; the modalities of `psf_generator.modalities` combine it with an illumination and a sample model into the image recorded by a given technique, starting with interferometric scattering microscopy (iSCAT).

## Image of a dipole

Consider a dipole of far-field amplitude $\mathbf{p}$ at the position $\mathbf{r}_p = (x_p, y_p, z_p)$ in the sample medium (index $n_s$), radiating

```{math}
    :label: eq:dipole-far-field

    \mathbf{E}(\mathbf{r}) = \left[(\hat{\mathbf{s}} \times \mathbf{p}) \times \hat{\mathbf{s}}\right] \frac{\mathrm{e}^{\mathrm{i} k_s r}}{r},
    \qquad k_s = \frac{2\pi n_s}{\lambda}.
```

The light crosses the coverslip ($n_g$, thickness $t_g$) and the immersion medium ($n_i$, thickness $t_i$), is collimated by an aplanatic objective of numerical aperture $\mathrm{NA}$ and focused on the camera by a tube lens of low NA, with a lateral magnification $M$.
Expressed in *object-space coordinates* $\boldsymbol{\rho}$ (camera coordinates divided by $M$), the image field is

```{math}
    :label: eq:image-field

    \hat{\mathbf{E}}(\boldsymbol{\rho}) = -\frac{\mathrm{i} k_i}{2\pi} \iint\limits_{s_x^2 + s_y^2 \leq s_{\max}^2}
    \frac{\tilde{\mathbf{e}}(\mathbf{s})}{\sqrt{\cos\theta}}\, \mathrm{e}^{\mathrm{i} k_i \mathbf{s}\cdot\boldsymbol{\rho}} \,\mathrm{d}s_x\,\mathrm{d}s_y
    = -\frac{\mathrm{i} k_i}{2\pi} \int_0^{\theta_{\max}}\!\!\mathrm{d}\theta \int_0^{2\pi}\!\!\mathrm{d}\phi\;
    \tilde{\mathbf{e}}(\theta, \phi)\, \sqrt{\cos\theta} \sin\theta\, \mathrm{e}^{\mathrm{i} k_i \mathbf{s}\cdot\boldsymbol{\rho}},
```

where $k_i = 2\pi n_i / \lambda$, $\theta$ is the polar angle in the immersion medium, $s_{\max} = \mathrm{NA}/n_i^*$ and $\tilde{\mathbf{e}}$ is the far field of the dipole *in the immersion medium*, rotated into the pupil plane by the objective.
Compared to the focusing integral [%s](#eq:cartesian-vectorial), four things change.

**Apodization.**
The objective maps the reference sphere onto a plane; energy conservation between the area elements $\mathrm{d}A_2 = f^2 \sin\theta\,\mathrm{d}\theta\,\mathrm{d}\phi$ on the sphere and $\mathrm{d}A_1 = f^2 \sin\theta \cos\theta\,\mathrm{d}\theta\,\mathrm{d}\phi$ on the plane gives the factor $1/\sqrt{\cos\theta}$, the inverse of the focusing apodization $\sqrt{\cos\theta}$ of {numref}`fig:correction-factors` (Eq. 73 of {cite:p}`foreman2011computational`).

**Jacobian.**
The tube lens has a low NA, so the Debye-Wolf integral in image space is a plain two-dimensional Fourier transform, without the $1/s_z$ factor of the focusing integral (Eq. 78 of {cite:p}`foreman2011computational`).
With the sine condition $n_i \sin\theta = M n_d \sin\theta_d$ the image-space variables map onto the object-space direction $\mathbf{s}$, up to the constant factor $\sqrt{n_i/n_d}\,/M$ that we drop: with $\mathrm{d}s_x\,\mathrm{d}s_y = \sin\theta\cos\theta\,\mathrm{d}\theta\,\mathrm{d}\phi$ the spherical form of Eq. [%s](#eq:image-field) has the weight $\sqrt{\cos\theta}\sin\theta$, exactly as the focusing integral.
This is the reciprocity between the excitation and detection PSFs: in a homogeneous medium the image of a dipole equals the focus field of the vectorial propagators with `apod_factor=True`, for the transverse components (the tube lens does not produce an axial component in the image).
The normalization of Eq. [%s](#eq:image-field) is such that (Parseval) $\iint |\hat{\mathbf{E}}|^2\,\mathrm{d}^2\rho = \iint |\mathbf{e}_i|^2\,\mathrm{d}\Omega$, the energy collected in the immersion medium.

**Fresnel coefficients.**
The dipole radiates in the sample, and its far field in the immersion medium is not simply the sample far field times the forward transmission coefficients $t_{s \to g}\, t_{g \to i}$.
Evaluating the transmitted angular spectrum by the method of stationary phase gives, for each polarization (Eqs. 10.36-10.38 of {cite:p}`Novotny_Hecht_2012`),

```{math}
    :label: eq:fresnel-reciprocity

    \mathbf{e}_i(\theta) = t_{s \to i}(\theta)\, \frac{n_i \cos\theta}{n_s \cos\theta_s}\, \mathbf{e}_s(\theta_s) = t_{i \to s}(\theta)\, \mathbf{e}_s(\theta_s),
    \qquad n_s \sin\theta_s = n_i \sin\theta,
```

i.e. the geometric factor $n_i\cos\theta / (n_s \cos\theta_s)$ turns the forward Fresnel coefficient into the coefficient of the *reverse* direction, which is the "dashed" coefficient of the vectorial ray tracing of {cite:p}`foreman2011computational`.
The factor is significant: for a dipole in water imaged with an oil objective it reaches $1.4$ at $\theta = 50^\circ$ and concentrates the collected light towards the critical angle.
It is the default of the imagers (`fresnel='reciprocal'`); `fresnel='forward'` reproduces the models of {cite:p}`mahmoodabadi2020point, dong2021fundamental` that use the forward coefficients alone.
Beyond the critical angle of the sample ($n_i \sin\theta > n_s$), $\cos\theta_s$ is imaginary: these are the evanescent components of the dipole field, collected when the dipole is close to the coverslip and decaying as $\mathrm{e}^{-k z_p \sqrt{n_i^2 \sin^2\theta - n_s^2}}$ (supercritical angle emission).

**Radiation pattern.**
The far field [%s](#eq:dipole-far-field) is decomposed into its $p$ (in the meridional plane) and $s$ components in the sample medium, $E_p = \mathbf{p}\cdot\hat{\boldsymbol{\theta}}_s$ with $\hat{\boldsymbol{\theta}}_s = (\cos\theta_s\cos\phi, \cos\theta_s\sin\phi, \sin\theta_s)$ and $E_s = \mathbf{p}\cdot\hat{\boldsymbol{\phi}}$, transmitted with $t_p$ and $t_s$, and rotated by the objective onto $(\cos\phi, \sin\phi, 0)$ and $(-\sin\phi, \cos\phi, 0)$.
With $A_0 = t_p\cos\theta_s + t_s$, $A_1 = t_p \sin\theta_s$ and $A_2 = t_p\cos\theta_s - t_s$, the pupil field is

```{math}
    \tilde{e}_x = \frac{p_x}{2}(A_0 + A_2\cos 2\phi) + \frac{p_y}{2} A_2 \sin 2\phi + p_z A_1 \cos\phi, \qquad
    \tilde{e}_y = \frac{p_x}{2} A_2 \sin 2\phi + \frac{p_y}{2}(A_0 - A_2\cos 2\phi) + p_z A_1 \sin\phi,
```

which reduces to the $q_i$ of {cite:p}`foreman2011computational` in a homogeneous medium ($A_0 = 1 + \cos\theta$, $A_1 = \sin\theta$, $A_2 = \cos\theta - 1$).
The $z$ axis points from the objective into the sample, so $z_p > 0$ is a dipole above the coverslip.

**Optical path.**
The stratified sample adds the phase $k\Lambda(\theta)$ of the path from the dipole to the objective relative to the design conditions (starred quantities), the same expression as the Gibson-Lanni factor [%s](#eq:stratified-layers-general-formula) with $t_s = z_p$:

```{math}
    :label: eq:imaging-opd

    \Lambda(\theta) = z_p \sqrt{n_s^2 - n_i^2\sin^2\theta} + t_i\, n_i \cos\theta
    + t_g \sqrt{n_g^2 - n_i^2\sin^2\theta} - t_g^* \sqrt{{n_g^*}^2 - n_i^2\sin^2\theta} - t_i^* \sqrt{{n_i^*}^2 - n_i^2\sin^2\theta}.
```

The immersion thickness follows from the axial position `z_focus` of the focal plane in the sample, $t_i = n_i (t_g^*/n_g^* + t_i^*/n_i^* - t_g/n_g - z_{\mathrm{focus}}/n_s)$, the paraxial focusing condition [%s](#eq:gibson-lanni-t_i) used by the propagators: a dipole at $z_p = z_{\mathrm{focus}}$ is in (paraxial) focus, and $t_i$ does not depend on the position of the dipole.
Aberrations of the detection path are added as Zernike modes on the pupil, like for the propagators.

### Spherical and Cartesian imagers

As for the propagators, the integral [%s](#eq:image-field) is evaluated either on a Cartesian pupil grid with a chirp Z transform (`CartesianDipoleImager`, any pupil aberration) or with the azimuthal integration done analytically (`SphericalDipoleImager`, axisymmetric aberrations only), which gives with $(\rho, \varphi)$ the polar coordinates of the pixel relative to the dipole

```{math}
    \hat{E}_x = -\mathrm{i} k_i \left[\frac{p_x}{2}(I_0 - I_2\cos 2\varphi) - \frac{p_y}{2} I_2 \sin 2\varphi + \mathrm{i} p_z I_1 \cos\varphi\right], \qquad
    \hat{E}_y = -\mathrm{i} k_i \left[-\frac{p_x}{2} I_2 \sin 2\varphi + \frac{p_y}{2}(I_0 + I_2\cos 2\varphi) + \mathrm{i} p_z I_1 \sin\varphi\right],
```

```{math}
    I_m(\rho) = \int_0^{\theta_{\max}} A_m(\theta)\, \mathrm{e}^{\mathrm{i} k \Lambda(\theta)}\, \mathrm{e}^{\mathrm{i} W(\theta)}\, \sqrt{\cos\theta}\sin\theta\, J_m(k_i \rho \sin\theta)\,\mathrm{d}\theta, \qquad m = 0, 1, 2.
```

A lateral displacement of the dipole is a shift of the image (a phase ramp $\mathrm{e}^{-\mathrm{i} k_i (s_x x_p + s_y y_p)}$ on the Cartesian pupil); its height $z_p$ only enters $\Lambda$, so all the heights of a dipole are computed in one batch.

## Modalities

A modality composes an illumination (the propagators, or a plane wave), a sample model and the detection path into the image recorded by a technique.
The first family of modalities is the coherent imaging of a Rayleigh scatterer with a plane-wave illumination: interferometric scattering microscopy (iSCAT), coherent bright-field microscopy (COBRI) and dark-field microscopy {cite:p}`taylor2019interferometric, mahmoodabadi2020point, dong2021fundamental, hitzelhammer2024unified`.

A particle of radius $a$ and permittivity $\epsilon_p$ has, in the sample medium ($\epsilon_s = n_s^2$), the Clausius-Mossotti polarizability $\alpha = 4\pi a^3 (\epsilon_p - \epsilon_s)/(\epsilon_p + 2\epsilon_s)$.
An incident plane wave of amplitude $E^0$ (x-polarized, normal incidence) that reaches the particle with the amplitude $t_{\mathrm{ill}} E^0$ induces a dipole whose far-field amplitude is $\mathbf{p} = k_s^2 \alpha\, t_{\mathrm{ill}} E^0\, \hat{\mathbf{x}} / (4\pi)$, and the scattered field at the camera $\mathbf{E}^{\mathrm{sca}}$ follows from Eq. [%s](#eq:image-field).
The camera records the interference with a reference wave,

```{math}
    :label: eq:iscat-intensity

    I(\boldsymbol{\rho}) = \left|\mathbf{E}^{\mathrm{ref}} + \mathbf{E}^{\mathrm{sca}}(\boldsymbol{\rho})\right|^2,
    \qquad C(\boldsymbol{\rho}) = \frac{I - |\mathbf{E}^{\mathrm{ref}}|^2}{|\mathbf{E}^{\mathrm{ref}}|^2},
```

where $C$ is the interferometric contrast, or interferometric PSF (iPSF).
The schemes differ by the reference and by the optical paths of the illumination:

| Scheme | Illumination at the particle | Reference $\mathbf{E}^{\mathrm{ref}}$ | Path of the illumination | Path of the reference |
|--------|------------------------------|---------------------------------------|--------------------------|-----------------------|
| iSCAT | $t_{ig} t_{gs} E^0$ through the objective | $\beta\, t_{ig}\, r_{gs}\, t_{gi}\, E^0$ reflected at the coverslip | $\Delta + n_s z_p$ | $2\Delta$ |
| COBRI | $E^0$ from the sample side | $\beta\, t_{sg}\, t_{gi}\, E^0$ transmitted | $-n_s z_p$ | $\Delta$ |
| dark-field | $t_{ig} t_{gs} E^0$ | $0$ | $\Delta + n_s z_p$ | -- |

Here $r_{gs} = (n_g - n_s)/(n_g + n_s)$ and $t_{ab} = 2n_a/(n_a + n_b)$ are the Fresnel coefficients at normal incidence, $\beta$ an optional attenuation of the reference (a partial reflector in the back focal plane) and $\Delta = (n_i t_i + n_g t_g) - (n_i^* t_i^* + n_g^* t_g^*)$ the excess single-pass path through the immersion medium and the coverslip with respect to the design conditions.
Together with $\Lambda(\theta)$ of Eq. [%s](#eq:imaging-opd), the phase of the scattered light relative to the reference is, for design coverslip and immersion,

```{math}
    \Lambda_{\mathrm{iSCAT}}(\theta) = n_s z_p (\cos\theta_s + 1) + n_i (t_i - t_i^*)(\cos\theta - 1), \qquad
    \Lambda_{\mathrm{COBRI}}(\theta) = n_s z_p (\cos\theta_s - 1) + n_i (t_i - t_i^*)(\cos\theta - 1),
```

the $\xi = \pm 1$ term of {cite:p}`dong2021fundamental`: the iSCAT signal oscillates with the particle height with the period $\lambda / 2 n_s$, whereas the COBRI signal does not.
Intensities are expressed in units of the incident intensity; the common factor $(n_i/n_d)/M^2$ is dropped.
With this normalization a 30 nm gold particle in water on a glass coverslip gives a contrast of several tens of percent in focus and a 100 kDa protein about $10^{-4}$, in line with mass photometry.

### Relation to published models

The model of {cite:p}`mahmoodabadi2020point`, used by {cite:p}`dong2021fundamental`, is the spherical imager with `fresnel='forward'`, i.e. without the geometric factor of Eq. [%s](#eq:fresnel-reciprocity), and with an ad hoc normalization of the scattered amplitude (a collection-efficiency factor $\eta$ and the power transmittance of the interface) where we use the exact far-field amplitude $k_s^2 \alpha / 4\pi$.
They also define the focus position $z_f$ through $t_i = t_i^* + (z_p - z_f) - n_i z_p / n_s$, which makes the immersion thickness depend on the particle height; our `z_focus` keeps $t_i$ a property of the microscope, $t_i = t_i^* - n_i z_{\mathrm{focus}}/n_s$, and both conventions coincide for a particle in focus ($z_p = z_f = z_{\mathrm{focus}}$).
The unified simulation platform of {cite:p}`hitzelhammer2024unified` computes the scattered far field of arbitrary particles with a boundary element method and images it with the same Richards-Wolf integral, weight $\sqrt{\cos\theta}\sin\theta$ and $\sqrt{n/n'}/M$ prefactor as Eq. [%s](#eq:image-field); the Rayleigh dipole of the modalities is its small-particle limit.

### Towards other modalities

The same building blocks describe scanning techniques.
For a confocal or image-scanning microscope, the excitation field $\mathbf{E}_{\mathrm{exc}}$ is the focus field of a propagator, the emission of a fluorophore at $\mathbf{r}$ (dipole $\mathbf{p} \propto \mathbf{E}_{\mathrm{exc}}(\mathbf{r})$ for a freely rotating molecule, or a fixed orientation) is imaged by a dipole imager, and the signal for a scan position $\boldsymbol{\rho}_s$ is the intensity integrated over the pinhole (confocal) or recorded by each element of the detector array (image scanning microscopy).
Such modalities only need a new subclass of `Modality` combining a propagator, an imager and a detector model; the JSON round trip, the registry and the position batching are inherited.
