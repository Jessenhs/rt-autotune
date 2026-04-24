use rt_autotune::AutoTune;

fn main() {
    nih_plug::wrapper::standalone::nih_export_standalone::<AutoTune>();
}
