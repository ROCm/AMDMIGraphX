#pragma once

#include "config.hpp"
#include "numpy.hpp"

#include <vector>
#include <algorithm>

using namespace mlinfer;

using NumpyVector = std::vector<std::vector<int64_t>>;

struct Dataset
{
    Dataset() = default;

    void initialize()
    {
        loadDataset();
        if (!_npy_files_loaded)
        {
            prepareSampleDataset();
        }
    }

    NumpyVector loadNumpy(npy::NpyFile& file)
    {
        NumpyVector numpyDataAll;
        auto load_size = file.GetTensorSize()/sizeof(int64_t);
        numpyDataAll.push_back(std::vector<int64_t>(load_size));
        file.LoadAll(numpyDataAll.back().data());

        NumpyVector numpyData;
        for(size_t i = 0; i < numpyDataAll.back().size(); i += SEQ_SIZE)
        {
            auto last = std::min(numpyDataAll.back().size(), i + SEQ_SIZE);
            numpyData.emplace_back(numpyDataAll.back().begin() + i, numpyDataAll.back().begin() + last);
        }

#ifdef TRACE
        for (auto& vec: numpyData)
        {
            std::cout << "Vector size: " << vec.size() <<  std::endl;
            for (auto val: vec)
            {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
#endif
        return numpyData;
    }

    size_t getLastIdx(int current_batch_idx) const
    {
        auto idx = _current_batch * BATCH_SIZE + current_batch_idx;
        auto res = std::find_if(std::rbegin(attention_mask[idx]), std::rend(attention_mask[idx]), [](uint64_t val) { return 1 == val;});
        size_t last_idx = std::distance(res, std::rend(attention_mask[idx]));
        #ifdef TRACE
        std::cout << "Last input idx: " << last_idx << std::endl;
        #endif
        return last_idx;
    }

    std::vector<int64_t> getInputIds()
    {
        std::vector<int64_t> inputIdsBatch;
        inputIdsBatch.reserve(SEQ_SIZE*BATCH_SIZE);
        for (size_t i = 0; i < BATCH_SIZE; ++i)
        {
            auto inputVec = input_ids[BATCH_SIZE*_current_batch + i];
            std::copy(inputVec.begin(), inputVec.end(), std::back_inserter(inputIdsBatch));
        }
        return inputIdsBatch;
    }

    std::vector<int64_t> getAttentionMask()
    {
        std::vector<int64_t> attentionMaskBatch;
        attentionMaskBatch.reserve(SEQ_SIZE*BATCH_SIZE);
        for (size_t i = 0; i < BATCH_SIZE; ++i)
        {
            auto attVec = attention_mask[BATCH_SIZE*_current_batch + i];
            std::copy(attVec.begin(), attVec.end(), std::back_inserter(attentionMaskBatch));
        }
        return attentionMaskBatch;
    }

    size_t size() const { return _size; }
    size_t currentBatchIdx() const { return _current_batch; }
    size_t batchNum() const {
        return _size / BATCH_SIZE + (_size % BATCH_SIZE != 0);
    }
    size_t getNext()
    {
        if (_current_batch < batchNum() - 1)
        {
            ++_current_batch;
        }
        #ifdef TRACE
        std::cout << "Current batch: " << _current_batch << std::endl;
        #endif
        return _current_batch;
    }

    Dataset(const Dataset &buf) = delete;
    Dataset &operator=(const Dataset &buf) = delete;
private:

    // e.g.: /dataset/input_ids_size_3_seq_256.npy
    std::string getDatasetPath(const std::string& datasetName)
    {
        std::stringstream path;
        path << DATASET_FOLDER << datasetName << "_size_" << std::to_string(DATASET_SIZE) << "_seq_" << std::to_string(SEQ_SIZE) << ".npy";
        return path.str();
    }

    void loadDataset()
    {
        std::string input_file_path = getDatasetPath("input_ids");
        std::string attention_mask_file_path = getDatasetPath("attention_mask");

        std::cout << "Input ids file: " << input_file_path << std::endl;
        std::ifstream input_file(input_file_path.c_str());
        std::ifstream attention_mask_file(attention_mask_file_path.c_str());
        if (input_file.good() && attention_mask_file.good())
        {
            npy::NpyFile input_ids_npy{input_file_path};
            npy::NpyFile attention_mask_npy{attention_mask_file_path};
            input_ids = loadNumpy(input_ids_npy);
            attention_mask = loadNumpy(attention_mask_npy);

            _size = input_ids.size();

            if (input_ids.size() == attention_mask.size())
            {
                std::cout << "Loaded numpy files\n";
                _npy_files_loaded = true;
            }
            else
            {
                std::cout << "Numpy files do not have the same size\n";
                input_ids.clear();
                attention_mask.clear();
            }
        }
        else
        {
            std::cout << "Unable to open numpy files\n";
        }
    }

    void prepareSampleDataset()
    {
        std::cout << "Numpy files are not loaded, using dummy data\n";
        std::vector<int64_t> input_ids_sample = {1,6804,338,5207,387,287,29973};
        input_ids_sample.resize(SEQ_SIZE, 0);
        std::vector<int64_t> attention_mask_sample = input_ids_sample;
        input_ids.emplace_back(std::move(input_ids_sample));
        std::transform(std::begin(attention_mask_sample), std::end(attention_mask_sample), std::begin(attention_mask_sample), [](auto i){
            return (i != 0) ? 1 : 0;
        });
        attention_mask.emplace_back(std::move(attention_mask_sample));

        _size = 1;
    }

    NumpyVector input_ids;
    NumpyVector attention_mask;

    size_t _size = 0;
    size_t _current_batch = 0;
    bool _npy_files_loaded = false;
};