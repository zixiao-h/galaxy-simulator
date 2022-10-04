from typing import NamedTuple
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define constants
G = 1.0

@nb.jit(nopython=True)
def verlet(timespan, dt, xv_initial, acceleration, args):
     """
     Acceleration function, must be jitted

     xv_initial is a (2, N, D) array of initial conditions; 2 for x and v, 
     N masses/particles, D dimensions

     args in the form args=(a, b, c, ...)
     """
     _, N, D = xv_initial.shape
     Nt = np.int(timespan/dt)
     t = np.linspace(0, timespan, Nt)
     x = np.zeros((Nt, N, D))
     v = np.zeros((Nt, N, D))

     x[0] = xv_initial[0]
     v[0] = xv_initial[1]

     # Warm up step to set first two positions and forces
     x[1] = x[0] + v[0]*dt
          
     # Loop through all points in time interval
     for i in range(1, Nt-1):
          F = acceleration(x[i], *args) # x is current position
          x[i+1] = 2*x[i] - x[i-1] + F*dt**2
          v[i] = (x[i+1]-x[i-1])/(2*dt)
          #print(i, ": ", x[i], v[i])
     v[Nt-1] = v[Nt-2]
     return (t, x, v)

@nb.jit(nopython=True)
def verlet_arr(timespan, dt, xv_initial, acceleration, args):
     """
     acceleration is (Nt, N, D) array of accelerations of each x element at each
     point in time. Must match Nt, N, D specified by timespan, dt, xv_initial

     xv_initial is a (2, N, D) array of initial conditions; 2 for x and v, 
     N masses/particles, D dimensions

     args in the form args=(a, b, c, ...)
     """
     _, N, D = xv_initial.shape
     Nt = np.int(timespan/dt)
     t = np.linspace(0, timespan, Nt)
     x = np.zeros((Nt, N, D))
     v = np.zeros((Nt, N, D))

     x[0] = xv_initial[0]
     v[0] = xv_initial[1]

     # Warm up step to set first two positions and forces
     x[1] = x[0] + v[0]*dt
          
     # Loop through all points in time interval
     for i in range(1, Nt-1):
          F = acceleration(x[i], i, *args) # x is current position
          x[i+1] = 2*x[i] - x[i-1] + F*dt**2
          v[i] = (x[i+1]-x[i-1])/(2*dt)
          #print(i, ": ", x[i], v[i])
     v[Nt-1] = v[Nt-2]
     return (t, x, v)

@nb.jit(nopython=True)
def p_acc(x, i, m_orbit, m_M):
    """
    Takes in array of positions of particles x, time index i, array of positions of 
    masses m_x, array of magnitudes of masses m_M
    Returns acceleration of each x element due to ALL masses as an array of same 
    size as the positions (i.e. acceleration of a test particle)
    """
    N, D = x.shape
    acc = np.zeros((N, D))
    
    for j in range(N):
        # Iterate over N particles
        r_x = x[j] - m_orbit[i] # Relative positions at ith instant in time
        r = np.sqrt((r_x*r_x).sum(axis=1))
        force = np.reshape(-G*m_M/(r**3+0.03), (1,-1))
        acc[j] = force @ r_x
    
    return acc

@nb.jit(nopython=True)
def m_acc(m_x, m_M):
    """
    Takes in array of positions of masses m_x, array of 
    magnitudes of masses m_M
    Returns acceleration of each x element due other masses as an array of same 
    size as the positions
    """
    N, D = m_x.shape
    acc = np.zeros((N, D))
    
    for i in range(N):
        individual_acc = np.zeros(D)
        for j in range(N):
            if not i == j:    
                r_x = m_x[i] - m_x[j] # Relative position
                r = np.sqrt((r_x*r_x).sum())
                individual_acc += -G*m_M[j]/r**3 * r_x
        acc[i] = individual_acc
    
    return acc

class mass:
    def __init__(self, x, v, M):
        """Initial position/velocity as np arrays"""
        self.x = x
        self.v = v
        self.M = M

class particle:
    def __init__(self, x, v):
        """Initial position and velocity vectors as np arrays"""
        self.x = x 
        self.v = v

class system:
    def __init__(self):
        self.particles = []
        self.masses = []
        self.p_t = particle(0, 0)
        self.m_t = mass(0,0,1)
    
    @property
    def m_t(self):
        return self._m_t
    @m_t.setter
    def m_t(self, m_t):
        self._m_t = m_t
    
    @property
    def p_t(self):
        return self._p_t
    @p_t.setter
    def p_t(self, p_t):
        self._p_t = p_t

    def particles_xv(self, i=-1):
        """
        Returns initial positions, velocities of particles as array
        No argument for both as [x, v], 0 for x and 1 for v only
        """
        if len(self.particles) == 0:
            return []
        x = [p.x for p in self.particles]
        v = [p.v for p in self.particles]
        if i == -1:
            return np.array([x, v])
        elif i == 0:
            return np.array(x)
        else:
            return np.array(v)
    
    def masses_xv(self, i=-1):
        """
        Returns initial positions, velocities of masses as array
        No argument for both as [x, v], 0 for x and 1 for v only
        """
        x = [mass.x for mass in self.masses]
        v = [mass.v for mass in self.masses]
        if i == -1:
            return np.array([x, v])
        elif i == 0:
            return np.array(x)
        else:
            return np.array(v)
    
    def masses_M(self):
        return np.array([mass.M for mass in self.masses])

    def add_particle(self, x, v):
        """Initial x, v as np arrays"""
        self.particles.append(particle(x, v))

    def add_mass(self, x, v, M):
        """Initial x, v as np arrays"""
        self.masses.append(mass(x, v, M))
        return self.masses[-1]

    def add_ring(self, r, n, mass):
        """
        Add a ring of n particles of radius r around some mass 
        in a circular orbit
        """
        # Applying gives velocity vector perpendicular to displacement from mass
        R = np.array([[0, -1], [1, 0]])
        # Calculate position/velocity of each particle and update
        for i in range(n):
            theta = i * 2*np.pi/n
            displacement = np.array([np.cos(theta), np.sin(theta)]) * r # from central mass
            x = displacement + mass.x
            v = R@displacement/r * np.sqrt(G * mass.M / r) + mass.v
            self.add_particle(x, v)

    def plot(self, lim=0):
        """Plot initial positions of all particles and masses"""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        particles_x = [particle.x[0] for particle in self.particles]
        particles_y = [particle.x[1] for particle in self.particles]
        masses_x = [mass.x[0] for mass in self.masses]
        masses_y = [mass.x[1] for mass in self.masses]
        ax.plot(particles_x, particles_y, 'o', ms=1, color='black')
        ax.plot(masses_x, masses_y, 'o', ms=3)
        ax.set_aspect('equal')
        if not lim == 0:
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

    def evolve(self, timespan, dt):
        """
        Returns t, particle data, masses data, data has shape 
        (particles, time, positions)
        """
        
        #N = int(timespan/dt) # Number of time points sampled
        # Evolve masses
        m_M = self.masses_M()
        initial_m = self.masses_xv()
        t, m_orbit, m_velocities = verlet(timespan, dt, initial_m, m_acc, args=(m_M,))
        self.m_t.x = m_orbit
        self.m_t.v = m_velocities
        self.m_t.M = self.masses_M()

        # Evolve particles
        if len(self.particles_xv()) != 0:
            initial_p = self.particles_xv()
            t, p_orbit, p_velocities = verlet_arr(timespan, dt, initial_p, p_acc, args=(m_orbit, m_M))
            self.p_t.x = p_orbit
            self.p_t.v = p_velocities

    def animate(self, filename, step, lim=0, xylims=0, fps_=30):
        """xylims as list of two tuples [(x1,x2), (y1,y2)]"""
        # Preparing data
        m_tx = self.m_t.x
        mx = m_tx[::step,:,0]
        my = m_tx[::step,:,1]
        if len(self.particles_xv()) != 0:
            p_tx = self.p_t.x
            px = p_tx[::step,:,0]
            py = p_tx[::step,:,1]
        else:
            px = np.zeros_like(mx)
            py = np.zeros_like(my)
        
        f = len(mx) # Number of frames
        
        # Prepare limits
        if lim == 0 and xylims == 0:
            print("set a limit")
            return 1
        elif xylims == 0:
            x1 = y1 = -lim
            x2 = y2 = lim
        elif lim == 0:
            x1, x2 = xylims[0]
            y1, y2 = xylims[1]

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(x1, x2), ylim=(y1, y2))
        ax.set_aspect('equal')

        particles_line, = ax.plot([], [], 'o', ms=1)
        masses_line, = ax.plot([], [], 'o', ms=3)

        def init():
            particles_line.set_data([], [])
            masses_line.set_data([], [])
            return particles_line, masses_line

        def animate_f(i):
            particles_line.set_data(px[i], py[i])
            masses_line.set_data(mx[i], my[i])
            return particles_line, masses_line

        ani = animation.FuncAnimation(fig, animate_f, np.arange(1, f),
                                      interval=25, blit=True, init_func=init)
        ani.save(filename, fps=fps_)
    
    def plot_t(self, lim=0):
        """
        Line plot of time evolution of system with optional limits
        """
        m_tx = self.m_t.x
        mx = m_tx[:,:,0]
        my = m_tx[:,:,1]
        plt.plot(mx, my, 'o-', lw=1, ms=1, color='blue')
        
        if len(self.particles_xv()) != 0:
            p_tx = self.p_t.x
            px = p_tx[:,:,0]
            py = p_tx[:,:,1]
            plt.plot(px, py, lw=1, color='black')
        
        if not lim == 0:
            plt.xlim(-lim,lim)
            plt.ylim(-lim,lim)
