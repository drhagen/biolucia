from unittest import TestCase
from collections import OrderedDict

from biolucia.model import Model
from biolucia.experiment import InitialValueExperiment
from biolucia.fitting import ActiveParameters, fit_parameters
from biolucia.observation import LinearWeightedSumOfSquaresObservation, AffineMeasurementUncertainty
from test_models import equilibrium_model, equilibrium_dose_variant


class ActiveParametersTestCase(TestCase):
    def test_extraction(self):
        m = equilibrium_model()
        con1 = InitialValueExperiment(equilibrium_dose_variant())
        con2 = InitialValueExperiment(Model().add('A0 = 15', 'kf = 0.3'))

        active_parameters = ActiveParameters.from_model_experiments(m, [con1], ['kf', 'kr'], [[]])
        self.assertEqual(active_parameters, ActiveParameters(OrderedDict([('kf', 0.5), ('kr', 0.2)]), [OrderedDict()]))

        active_parameters = ActiveParameters.from_model_experiments(m, [con1, con2], ['kf', 'kr'], [])
        self.assertEqual(active_parameters, ActiveParameters(OrderedDict([('kf', 0.5), ('kr', 0.2)]),
                                                             [OrderedDict(), OrderedDict()]))

        active_parameters = ActiveParameters.from_model_experiments(m, [con1, con2], ['kr', 'kf'], [])
        self.assertEqual(active_parameters, ActiveParameters(OrderedDict([('kr', 0.2), ('kf', 0.5)]),
                                                             [OrderedDict(), OrderedDict()]))

        active_parameters = ActiveParameters.from_model_experiments(m, [con1, con2], ['kr', 'kf'], [])
        self.assertEqual(active_parameters, ActiveParameters(OrderedDict([('kr', 0.2), ('kf', 0.5)]),
                                                             [OrderedDict(), OrderedDict()]))

        active_parameters = ActiveParameters.from_model_experiments(m, [con1, con2], [], [[], ['A0']])
        self.assertEqual(active_parameters, ActiveParameters(OrderedDict(),
                                                             [OrderedDict(), OrderedDict([('A0', 15)])]))

        active_parameters = ActiveParameters.from_model_experiments(m, [con1, con2], [], [[], ['A0', 'kf']])
        self.assertEqual(active_parameters, ActiveParameters(OrderedDict(),
                                                             [OrderedDict(), OrderedDict([('A0', 15), ('kf', 0.3)])]))

    def test_updating(self):
        m = equilibrium_model()
        con1 = InitialValueExperiment(equilibrium_dose_variant())
        con2 = InitialValueExperiment(Model().add('A0 = 15', 'kf = 0.3'))

        active_parameters = ActiveParameters.from_model_experiments(m, [con1, con2], ['kr'], [[], ['A0', 'kf']])
        updated_parameters = active_parameters.update_parameters([4.0, 1.3, 2.0])
        self.assertEqual(updated_parameters, ActiveParameters(OrderedDict([('kr', 4.0)]),
                                                              [OrderedDict(), OrderedDict([('A0', 1.3), ('kf', 2.0)])]))

        active_parameters = ActiveParameters.from_model_experiments(m, [con2, con2], ['kr'], [['kf'], ['A0', 'kf']])
        updated_parameters = active_parameters.update_parameters([4.0, 3.4, 1.3, 2.0])
        self.assertEqual(updated_parameters,
                         ActiveParameters(OrderedDict([('kr', 4.0)]),
                                          [OrderedDict([('kf', 3.4)]), OrderedDict([('A0', 1.3), ('kf', 2.0)])]))


class FittingTestCase(TestCase):
    def test_basic_fitting(self):
        m = Model().add('k = 10')
        con = InitialValueExperiment()
        obj = LinearWeightedSumOfSquaresObservation([0], ['k'], AffineMeasurementUncertainty(1, 0)).objective([20])

        fitted_parameters = fit_parameters(m, [con], [obj], model_parameters='k')

        self.assertAlmostEqual(fitted_parameters.model_parameters['k'], 20)
