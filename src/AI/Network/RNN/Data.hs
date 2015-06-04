-----------------------------------------------------------------------------
--
-- Module      :  AI.Network.RNN.Data
-- Copyright   :  (c) JP Moresmau
-- License     :  BSD3
--
-- Maintainer  :  JP Moresmau <jp@moresmau.fr>
-- Stability   :  experimental
-- Portability :
--
-- |
--
-----------------------------------------------------------------------------

module AI.Network.RNN.Data  where

import qualified Data.Text as T
import qualified Data.Map as DM
import Data.Maybe
import Data.Tuple
import Data.List
import Data.Ord
import Control.Monad.Random as R
import Control.Monad
import Numeric.LinearAlgebra.HMatrix  as M
import Data.Foldable as F (toList)

import AI.Network.RNN.RNN

textToTrainData :: T.Text -> ([Vector Double], [Vector Double],DM.Map Int Char)
textToTrainData t =
    let (ids,m) = T.foldl' toIDs ([],DM.empty) t
        sz = DM.size m
        is = map (toArr sz) $ reverse ids
    in (is,tail is ++ [M.fromList $ replicate sz 0],DM.fromList $ map swap $ DM.assocs m)
    where
        toIDs :: ([Int],DM.Map Char Int) -> Char -> ([Int],DM.Map Char Int)
        toIDs (ids,m) c =
            let mmyid = DM.lookup c m
            in case mmyid of
                Just myid -> (myid:ids,m)
                Nothing   ->
                    let myid = DM.size m
                    in (myid:ids,DM.insert c myid m)
        toArr :: Int -> Int -> Vector Double
        toArr sz idx = M.fromList $ (replicate idx 0) ++ [1] ++ (replicate (sz-idx-1) 0)

dataToText :: DM.Map Int Char -> [Vector Double] -> T.Text
dataToText m  = T.pack . F.toList . fmap toC
    where
        toC :: Vector Double -> Char
        toC ds = fromJust $ DM.lookup (maxIndex ds) m
        --maxIndex =fst . maximumBy (comparing snd) . zip [0..]

randDataToText :: RandomGen g => DM.Map Int Char -> [Vector Double] -> Rand g T.Text
randDataToText m  = (liftM T.pack) . mapM (toC . M.toList)
    where
        toC :: RandomGen g => [Double] -> Rand g Char
        toC = R.fromList . map (\(ix,d)->(fromJust $ DM.lookup ix m,toRational d)) . zip [0..]

generate :: RandomGen g => DM.Map Int Char -> Int -> Vector Double -> RNNetwork -> Rand g T.Text
generate m nb is rnn = do
    let (_,_,alls) = foldl' go (rnn,is,[is]) [1..nb-1]
    randDataToText m $ alls
    where
        go :: (RNNetwork,Vector Double,[Vector Double]) -> Int -> (RNNetwork,Vector Double,[Vector Double])
        go (rnn1,is1,oss) _ =
            let (rnn2,os) = evalStep rnn1 is1
            in (rnn2,os,oss ++ [os])
