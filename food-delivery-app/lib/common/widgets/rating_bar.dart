import 'package:flutter/material.dart';
import 'package:food_delivery_app/utils/constants/colors.dart';
import 'package:food_delivery_app/utils/constants/sizes.dart';

class CRatingBar extends StatelessWidget {
  final String? prefixText;
  final double value;
  const CRatingBar({
    this.prefixText,
    required this.value,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        if(prefixText != null)...[
          Text('$prefixText'),
          SizedBox(width: TSize.spaceBetweenItemsHorizontal,)
        ],
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(TSize.borderRadiusMd),
            child: SizedBox(
              height: TSize.sm,
              child: LinearProgressIndicator(
                value: value,
                color: TColor.primary,
                backgroundColor: Colors.grey[300],
              ),
            ),
          ),
        ),
      ],
    );
  }
}