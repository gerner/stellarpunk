#ifndef MARKOV_H
#define MARKOV_H

#include <cassert>
#include <vector>
#include <string>
#include <deque>
#include <unordered_map>
#include <random>
#include <sstream>
#include <fstream>
#include <iostream>
#include <thread>

#include<boost/tokenizer.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#define TOKEN_ID_T uint16_t
#define TOKEN_START 0
#define TOKEN_END 1


template<size_t N>
struct NGram {
    TOKEN_ID_T tokens[N];

    NGram() {
    }
    NGram(const NGram& other) {
        for(size_t i=0; i<N; i++) {
            tokens[i] = other.tokens[i];
        }
    }

    bool operator==(NGram<N> const& rhs) const {
        for(size_t i=0; i<N; i++) {
            if(tokens[i] != rhs.tokens[i]) {
                return false;
            }
        }
        return true;
    }
};

template<size_t N>
struct NGramHash {
    std::size_t operator()(const NGram<N>& s) const noexcept
    {
        std::size_t hash = 5381;
        int c;

        for(size_t i=0; i<N; i++) {
            c = s.tokens[i];
            hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
        }

        return hash;
    }
};

template<size_t N>
void add_count(
        std::unordered_map<NGram<N>, std::unordered_map<TOKEN_ID_T, size_t>, NGramHash<N> >& ngram_counts,
        const NGram<N>& ngram_prefix,
        const TOKEN_ID_T& token_id) {
    //adds a count for token_id for the prefix ngram_prefix in ngram_counts

    if(ngram_counts.count(ngram_prefix) == 0) {
        ngram_counts[ngram_prefix] = std::unordered_map<TOKEN_ID_T, size_t>();
    }
    if(ngram_counts[ngram_prefix].count(token_id) == 0) {
        ngram_counts[ngram_prefix][token_id] = 1;
    } else {
        ngram_counts[ngram_prefix][token_id] += 1;
    }
    return;

    /*
    // first find the right set of counts
    const auto& token_counts_itr = ngram_counts.find(ngram_prefix);
    if(token_counts_itr == ngram_counts.end()) {
        ngram_counts.insert(std::make_pair(ngram_prefix, std::unordered_map<TOKEN_ID_T, size_t>())).first->second.insert(std::make_pair(token_id, 1));
        return;
    }
    const auto& current_count_itr = token_counts_itr->second.find(token_id);
    if(current_count_itr == token_counts_itr->second.end()) {
        token_counts_itr->second.insert(std::make_pair(token_id, 1));
    } else {
        current_count_itr->second += 1;
    }*/
}

template<size_t N>
NGram<N> k(const std::deque<TOKEN_ID_T>& ngram) {
    assert(ngram.size() == N);
    NGram<N> key;
    size_t i = 0;
    for(const TOKEN_ID_T& token_id : ngram) {
        key.tokens[i++] = token_id;
    }
    return key;
}

template<size_t N, class T=std::string>
class MarkovModel {
private:
    // distinct tokens, index is the token id
    std::vector<T> tokens_;
    // map from ngram prefix to token id array and prob dist for those tokens
    std::unordered_map<NGram<N-1>, std::pair<std::vector<TOKEN_ID_T>, std::vector<size_t> >, NGramHash<N-1> > token_counts_;

public:
    bool train_from_file(std::string filename) {
        std::ifstream file(filename, std::ios_base::in | std::ios_base::binary);
        if(file.fail()) {
            return false;
        }
        boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
        in.push(boost::iostreams::gzip_decompressor());
        in.push(file);
        std::istream istream(&in);
        train(istream);
        return true;
    }

    template<class IStream>
    void train(IStream& in) {
        tokens_.clear();
        token_counts_.clear();

        std::unordered_map<T, TOKEN_ID_T> token_ids;
        std::unordered_map<NGram<N-1>, std::unordered_map<TOKEN_ID_T, size_t>, NGramHash<N-1> > ngram_counts;

        // reserve two spots for TOKEN_START and TOKEN_END
        tokens_.push_back(T());
        tokens_.push_back(T());

        // TODO: parametrize the tokenizer for extensibility
        // maybe a template parameter?
        int offsets[1] = {1};
        boost::offset_separator f(offsets, offsets+1);
        //size_t examples_so_far = 0;
        for (std::string line; std::getline(in, line);) {
            //if(examples_so_far++ % 100000 == 0) {
            //    std::cerr << ".";
            //}
            // prep an ngram prefix with TOKEN_START
            std::deque<TOKEN_ID_T> ngram_prefix;
            for(size_t i=0; i<N-1; i++){
                ngram_prefix.push_back(TOKEN_START);
            }

            boost::tokenizer<boost::offset_separator> tokens(line, f);
            for(const T& token : tokens) {
                // get a token id (and possibly add the token to tokens)
                const auto& token_id_itr = token_ids.find(token);
                TOKEN_ID_T token_id;
                if(token_id_itr == token_ids.end()) {
                    token_id = token_ids.insert(std::make_pair(token, tokens_.size())).first->second;
                    tokens_.push_back(token);
                } else {
                    token_id = token_id_itr->second;
                }

                // add a count for this token following ngram prefix
                add_count(ngram_counts, k<N-1>(ngram_prefix), token_id);

                // pop left token from prefix
                ngram_prefix.pop_front();
                // push right current token
                ngram_prefix.push_back(token_id);
            }
            // add a count for the end token
            add_count(ngram_counts, k<N-1>(ngram_prefix), TOKEN_END);
        }
        //std::cerr << "done with examples." << std::endl;

        // compute probabilities
        // for each ngram prefix
        for(const auto& itr : ngram_counts) {
            std::vector<TOKEN_ID_T> token_options;
            std::vector<size_t> counts;
            for(const auto& itr2 : itr.second) {
                token_options.push_back(itr2.first);
                counts.push_back(itr2.second);
            }
            token_counts_[itr.first] = std::make_pair(std::vector<TOKEN_ID_T>(token_options), std::vector<size_t>(counts));
        }

        //std::cerr << "done prepping counts." << std::endl;
    }

    std::string generate(uint32_t seed) {
        // source of randomness, seeded externally
        std::mt19937 gen(seed);
        std::deque<T> result;

        // prep an ngram prefix with TOKEN_START
        std::deque<TOKEN_ID_T> ngram_prefix;
        for(size_t i=0; i<N-1; i++){
            ngram_prefix.push_back(TOKEN_START);
        }

        // choose a token according to dist for current prefix
        TOKEN_ID_T token_id;
        {
            if(token_counts_.count(k<N-1>(ngram_prefix)) == 0) {
                return std::string("");
            }
            const std::pair<std::vector<TOKEN_ID_T>, std::vector<size_t> >& token_info = token_counts_[k<N-1>(ngram_prefix)];
            size_t index = std::discrete_distribution<>(token_info.second.begin(), token_info.second.end())(gen);
            token_id = token_info.first[index];
        }

        // while token is not TOKEN_END
        while(token_id != TOKEN_END) {
            result.push_back(tokens_[token_id]);

            // move ngram_prefix forward
            ngram_prefix.pop_front();
            ngram_prefix.push_back(token_id);

            // choose a token according to dist for current prefix
            const auto& token_info = token_counts_.find(k<N-1>(ngram_prefix));
            assert(token_info != token_counts_.end());
            size_t index = std::discrete_distribution<>(token_info->second.second.begin(), token_info->second.second.end())(gen);
            token_id = token_info->second.first[index];
        }

        // post process the sequence of tokens
        // return the result
        std::ostringstream imploded;
        std::copy(result.begin(), result.end(),
           std::ostream_iterator<std::string>(imploded, ""));
        return imploded.str();
    }

    bool save_to_file(std::string filename) {
        std::ofstream file(filename, std::ios_base::out | std::ios_base::binary);
        if(file.fail()) {
            return false;
        }
        boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
        out.push(boost::iostreams::gzip_compressor());
        out.push(file);
        std::ostream ostream(&out);
        save(ostream);
        return true;
    }

    template<class OStream>
    void save(OStream& out) {
        // save N
        size_t n = N;
        out.write(reinterpret_cast<char*>(&n), sizeof(size_t));

        // save the list of tokens, starting with a count
        size_t token_count = tokens_.size();
        out.write(reinterpret_cast<char*>(&token_count), sizeof(size_t));
        for(std::string token : tokens_) {
            // write each token, including the null
            out.write(token.c_str(), token.size()+1);
        }

        // save the ngram counts starting with a count of prefixes
        size_t ngram_prefix_count = token_counts_.size();
        out.write(reinterpret_cast<char*>(&ngram_prefix_count), sizeof(size_t));
        for(const auto& itr : token_counts_) {
            // save the ngram prefix
            for(size_t i=0; i<N-1; i++) {
                TOKEN_ID_T token_id = itr.first.tokens[i];
                out.write(reinterpret_cast<char*>(&token_id), sizeof(TOKEN_ID_T));
            }
            assert(itr.second.first.size() == itr.second.second.size());
            // save number of tokens
            size_t token_id_count = itr.second.first.size();
            out.write(reinterpret_cast<char*>(&token_id_count), sizeof(size_t));
            // save the token ids
            for(const TOKEN_ID_T& token_id : itr.second.first) {
                out.write(reinterpret_cast<const char*>(&token_id), sizeof(TOKEN_ID_T));
            }
            // save the counts
            for(const size_t& count : itr.second.second) {
                out.write(reinterpret_cast<const char*>(&count), sizeof(size_t));
            }
        }
    }

    bool load_from_file(std::string filename) {
        std::ifstream file(filename, std::ios_base::in | std::ios_base::binary);
        if(file.fail()) {
            return false;
        }
        boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
        in.push(boost::iostreams::gzip_decompressor());
        in.push(file);
        std::istream istream(&in);
        load(istream);
        return true;
    }

    template<class IStream>
    void load(IStream& in) {
        tokens_.clear();
        token_counts_.clear();

        // read N
        size_t n;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        assert(n == N);

        // read tokens, starting with a count
        size_t token_count;
        in.read(reinterpret_cast<char*>(&token_count), sizeof(token_count));
        for(size_t i=0; i<token_count; i++) {
            std::string token;
            std::getline(in, token, '\0');
            tokens_.push_back(token);
        }

        // read ngram counts, starting with a count of prefixes
        size_t prefix_count;
        in.read(reinterpret_cast<char*>(&prefix_count), sizeof(prefix_count));
        for(size_t i=0; i<prefix_count; i++) {
            // read ngram prefix
            NGram<N-1> ngram_prefix;
            for(size_t j=0; j<N-1; j++) {
                in.read(reinterpret_cast<char*>(&(ngram_prefix.tokens[j])), sizeof(TOKEN_ID_T));
            }

            // read number of tokens
            size_t token_id_count;
            in.read(reinterpret_cast<char*>(&token_id_count), sizeof(size_t));

            // read token ids
            std::vector<TOKEN_ID_T> token_ids;
            for(size_t j=0; j<token_id_count; j++) {
                TOKEN_ID_T token_id;
                in.read(reinterpret_cast<char*>(&token_id), sizeof(TOKEN_ID_T));
                token_ids.push_back(token_id);
            }

            // read counts
            std::vector<size_t> counts;
            for(size_t j=0; j<token_id_count; j++) {
                size_t count;
                in.read(reinterpret_cast<char*>(&count), sizeof(size_t));
                counts.push_back(count);
            }

            token_counts_[ngram_prefix] = std::make_pair(token_ids, counts);
        }
    }
};

typedef MarkovModel<1> MarkovModel1;
typedef MarkovModel<2> MarkovModel2;
typedef MarkovModel<3> MarkovModel3;
typedef MarkovModel<4> MarkovModel4;
typedef MarkovModel<5> MarkovModel5;
typedef MarkovModel<6> MarkovModel6;
typedef MarkovModel<7> MarkovModel7;

void train_from_file(MarkovModel5* model, std::string filename) {
    model->train_from_file(filename);
}

void load_many_models(std::vector<MarkovModel5*> models, std::vector<std::string> filenames) {
    assert(models.size() == filenames.size());
    //std::vector<std::thread> threads;
    for(size_t i=0; i<models.size(); i++) {
        //threads.push_back(std::thread(train_from_file, models[i], filenames[i]));
        train_from_file(models[i], filenames[i]);
    }
    //for(size_t i=0; i<threads.size(); i++) {
    //    threads[i].join();
    //}
}

#endif
