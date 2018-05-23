import numpy
import numpy.random
import time

time_it = False

import scipy
import weave    # scipy.weave for python 2.6; weave for python 2.7
                # WARNING!!! dt in python 3 should be ((float)dt)) for c++ code

def poisson_generator(rate,tsim):

    isi = numpy.random.exponential(1.0/rate,int(tsim*2.0*rate))

    spikes = numpy.add.accumulate(isi)

    i=numpy.searchsorted(spikes,tsim)

    return numpy.resize(spikes,(i,))


def shot_noise_generator(dt,tau,q,nu,tsim):
    """ generates shot noise using forward euler
    dt = resolution
    tau = decay time
    q = shot jump
    nu = rate of poisson process
    tsim = how long to simulate
    """

    if time_it:
        t1 = time.time()

    st = poisson_generator(nu,tsim)
    t = numpy.arange(0,tsim,dt)
    y = numpy.zeros(numpy.shape(t),float)

    y[0] = 0.0

    # use weave 

    code = """

    int j = 0;

    double val = exp(-((float)dt)/tau);

    for(int i=1;i<Ny[0];i++) {

      // decay
      y(i) = y(i-1)*val;

      // spikes?
      while (t(i)>st(j)) {
        y(i)+=(q*exp( (st(j)-t(i))/tau ) );
        j++;
      }
    
    }
    """

    #scipy.weave.inline(code,['y', 'dt', 'tau', 'q', 'st','t'],
    #             type_converters=scipy.weave.converters.blitz)
    weave.inline(code, ['y', 'dt', 'tau', 'q', 'st', 't'],
                       type_converters=weave.converters.blitz)
      

    if time_it:
        print 'Elapsed ',time.time()-t1,' seconds.'
        
    return (y,t)




def OU_generator_weave(dt,tau,sigma,y0,tsim):

    """ generates an OU process using forward euler
    dt = resolution
    tau = correlation time
    sigma = std dev of process
    y0 = mean/initial value
    tsim = how long to simulate
    """

    if time_it:
        t1 = time.time()


    t = numpy.arange(0,tsim,dt)
    y = numpy.zeros(numpy.shape(t),float)
    rng = numpy.random.normal(size=numpy.shape(t))

    y[0] = y0

    # python loop... bad+slow!
    #for i in xrange(1,len(t)):
    #    y[i] = y[i-1]+dt/tau*(y0-y[i-1])+numpy.sqrt(2*dt/tau)*sigma*numpy.random.normal()

    # use weave instead

    code = """

    double sig = (double)sigma;
    double val = sqrt(2*((float)dt)/tau)*sig;
    
    for(int i=1;i<Ny[0];i++) {
      y(i) = y(i-1) +((float)dt)/tau*(y0-y(i-1))+val*rng(i-1);
    }
    """

    #scipy.weave.inline(code,['y', 'dt', 'tau', 'sigma', 'rng','y0'],
    #             type_converters=scipy.weave.converters.blitz)
    weave.inline(code, ['y', 'dt', 'tau', 'sigma', 'rng', 'y0'],
                       type_converters=weave.converters.blitz)
      

    if time_it:
        print 'Elapsed ',time.time()-t1,' seconds.'
        
    return (y,t)
    

def v_solver(dt,veff,gtot,p):
    """
    solves for the membrane potential
    (with spiking dynamics, neuron parameters in p)

    given veff,gtot:

    dv/dt = gtot(t)/p.Cm(veff(t)-v)
    
    

    """

    if time_it:
        t1 = time.time()

    v = numpy.zeros(numpy.shape(veff),float)

    if numpy.shape(gtot)!=numpy.shape(veff):
        raise Exception,'gtot and veff must be of the same shape'

    v[0] = p.vr

    cm = p.Cm
    vth = p.vth
    vr = p.vr
    a = []

    # use weave

    code = """
    for(int i=1;i<Nv[0];i++) {
      v(i) = (v(i-1)-veff(i-1))*exp(-((float)dt)*gtot(i-1)/cm)+veff(i-1);
      if (v(i)>vth) {
        v(i) = vr;
        a.append(i);
      }
    }
    """

    #scipy.weave.inline(code,['v', 'gtot', 'dt', 'veff', 'cm', 'vr', 'vth','a'],
    #             type_converters=scipy.weave.converters.blitz)
    weave.inline(code, ['v', 'gtot', 'dt', 'veff', 'cm', 'vr', 'vth', 'a'],
                       type_converters=weave.converters.blitz)
      

    if time_it:
        print 'Elapsed ',time.time()-t1,' seconds.'
        
    return (v,a)
    


def OU_generator_threshold(dt,tau,sigma,y0,tsim,vth,vr):
    """ generates an OU process with threshold and reset
    using forward euler
    dt = resolution
    tau = correlation time
    sigma = std dev of process
    y0 = mean/initial value
    tsim = how long to simulate
    vth = threshold
    vr = reset
    """

    if time_it:
        t1 = time.time()


    t = numpy.arange(0,tsim,dt)
    y = numpy.zeros(numpy.shape(t),float)
    rng = numpy.random.normal(size=numpy.shape(t))

    y[0] = y0

    a = []

    # python loop... bad+slow!
    #for i in xrange(1,len(t)):
    #    y[i] = y[i-1]+dt/tau*(y0-y[i-1])+numpy.sqrt(2*dt/tau)*sigma*numpy.random.normal()

    # use weave instead

    code = """

    double sig = (double)sigma;
    double val = sqrt(2*((float)dt)/tau)*sig;
    
    for(int i=1;i<Ny[0];i++) {
      y(i) = y(i-1) +((float)dt)/tau*(y0-y(i-1))+val*rng(i-1);
      if (y(i)>vth) {
        y(i) = vr;
        a.append(i);
      }
    }
    """

    #scipy.weave.inline(code,['y', 'dt', 'tau', 'sigma', 'rng','y0','vth','vr','a'],
    #             type_converters=scipy.weave.converters.blitz)
    weave.inline(code, ['y', 'dt', 'tau', 'sigma', 'rng', 'y0', 'vth', 'vr', 'a'],
                       type_converters=weave.converters.blitz)
      

    if time_it:
        print 'Elapsed ',time.time()-t1,' seconds.'
        
    return (y,a)
    


def OU_generator_slow(dt,tau,sigma,y0,tsim):

    """ generates an OU process using forward euler
    dt = resolution
    tau = correlation time
    sigma = std dev of process
    y0 = mean/initial value
    tsim = how long to simulate
    """

    if time_it:
        t1 = time.time()


    t = numpy.arange(0,tsim,dt)
    y = numpy.zeros(numpy.shape(t),float)

    y[0] = y0

    # python loop... bad+slow!
    for i in xrange(1,len(t)):
        y[i] = y[i-1]+dt/tau*(y0-y[i-1])+numpy.sqrt(2*dt/tau)*sigma*numpy.random.normal()

    if time_it:
        print 'Elapsed ',time.time()-t1,' seconds.'
        
    return (y,t)


def OU_generator_slow1(dt,tau,sigma,y0,tsim):

    """ generates an OU process using forward euler
    dt = resolution
    tau = correlation time
    sigma = std dev of process
    y0 = mean/initial value
    tsim = how long to simulate
    """

    if time_it:
        t1 = time.time()


    t = numpy.arange(0,tsim,dt)
    w = numpy.random.normal(size=numpy.shape(t))
    y = numpy.zeros(numpy.shape(t),float)

    y[0] = y0

    # python loop... bad+slow!
    for i in xrange(1,len(t)):
        y[i] = y[i-1]+dt/tau*(y0-y[i-1])+numpy.sqrt(2*dt/tau)*sigma*w[i]

    if time_it:
        print 'Elapsed ',time.time()-t1,' seconds.'

        
    return (y,t)

OU_generator = OU_generator_weave
