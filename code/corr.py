import numpy
import scipy
import scipy.weave



def autocorr_slow(x,n):

    corr = numpy.zeros((n,),float)
    

    for i in xrange(1,n):

        corr[i] = sum(x[:-i]*x[i:])

    return corr



def autocorr_weave(x,n):

    corr = numpy.zeros((n,),float)

    nu = numpy.sqrt(numpy.dot(x,x))

    code = """

    double sum;
    int j;

    for(int i=0;i<Ncorr[0];i++) {

      sum = 0.0;

      for(j=0;j<Nx[0]-i;j++) {

        sum+=x(j)*x(j+i);

      }

      corr(i) = sum/(Nx[0]-i);

    }

    """

    scipy.weave.inline(code,['corr', 'x','nu'],
                 type_converters=scipy.weave.converters.blitz)


    return corr


def myacorr(x,n):

    corr = numpy.zeros((n,),float)

    nu = numpy.sqrt(numpy.dot(x,x))

    code = """

    double sum;
    int j;

    for(int i=0;i<Ncorr[0];i++) {

      sum = 0.0;

      for(j=0;j<Nx[0]-i;j++) {

        sum+=x(j)-x(j+i);

      }

      corr(i) = sum/(Nx[0]-i);

    }

    """                                                                                                

    scipy.weave.inline(code,['corr', 'x','nu'],
                 type_converters=scipy.weave.converters.blitz)


    return corr
