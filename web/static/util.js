function to_mmss(s) {
    mm = Math.floor(s / 60);
    ss = s - mm * 60;
    return `${mm} : ${ss.toFixed(2)}`
}

function sorted_index(array, value) {
    var low = 0,
        high = array.length;

    while (low < high) {
        var mid = (low + high) >>> 1;
        if (array[mid] < value) low = mid + 1;
        else high = mid;
    }
    return low;
}