import flatbuffer_reader as reader
import os
import csv
import sys


def get_weights_size(model):
    size = 0
    weights_num = 0
    for weight_type in model.weight_types:
        totals = model.weight_types[weight_type]
        if totals['weights'] > 0:
            size += totals['bytes']
            weights_num += totals['weights']

    return weights_num, size


def read_model(file_name, GRAPH_INDEX=0):
    tflite_file = None
    try:
        tflite_file = open(file_name, 'rb')
    except IOError:
        print("Failed to open file \"%s\"." % file_name)
        quit()

    print("=" * (len(file_name) + 14 + 21))
    print("====== Reading flatbuffer \"%s\" ======" % file_name)
    print("=" * (len(file_name) + 14 + 21))
    flatbuffer = tflite_file.read()
    print("Model Read Successful")

    base_name = file_name
    if base_name[-7:] == ".tflite":
        base_name = base_name[:-7]

    return reader.AnalysedTFliteModel(flatbuffer, GRAPH_INDEX)


def main(argv):
    # Get Weights Size
    tflite_file = sys.argv[1]
    if not tflite_file.endswith('.tflite'):
        tflite_file = tflite_file + '.tflite'

    file_name = 'model/' + tflite_file
    model = read_model(file_name, GRAPH_INDEX=0)
    weights_num, size = get_weights_size(model)

    # Generate TF Benchmark CSV
    outdir = 'output'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    output_report = outdir + '/report.csv'
    cmd = 'tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model --graph=model/car_sensor_reduced_drange.tflite --enable_op_profiling=true --profiling_output_csv_file="' + output_report + '"'
    os.system(cmd)

    # Append memory rows
    with open(output_report, mode='r') as report:
        csv_reader = csv.reader(report, delimiter=',')
        memory_rows = [['Memory Summary:'],
                       ['Number of Weights:', str(weights_num)],
                       ['Weights Size (KB):', str(size)],
                       ['=================================================================='],
                       []]
        rows = []
        for row in memory_rows:
            rows.append(row)
        for row in csv_reader:
            rows.append(row)

    with open(output_report, mode='w') as report:
        csv_writer = csv.writer(report, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for row in rows:
            csv_writer.writerow(row)

    print("\nReport generated in "+output_report)


if __name__ == "__main__":
    main(sys.argv[1:])
